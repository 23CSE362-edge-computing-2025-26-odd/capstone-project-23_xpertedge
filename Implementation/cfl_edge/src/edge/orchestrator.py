# FILE: cfl_edge/src/edge/orchestrator.py

from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import csv
import time
import torch
import copy
import numpy as np

# --- CORRECTED IMPORTS ---
from cfl_edge.src.edge.configs import load_sim_config, SimConfig, SimDirs
from cfl_edge.src.edge.clients import build_clients_from_sizes, ClientProfile
from cfl_edge.src.edge.latency import estimate_times_for_selected
from cfl_edge.src.utils.io_json import read_json, write_json
from ci.src.models import CNN
from ci.src.genetic import Hyperparameters
from cfl_edge.src.DeCFL.cluster import compute_pairwise_dis, adaptive_kmeans
from cfl_edge.src.edge.scheduler_advanced import select_clients_advanced
from cfl_edge.src.edge.scheduler_baseline import select_random, build_aggregation_sets

def _read_shared_inputs(cfg: SimConfig) -> tuple[Dict[str, int], int]:
    cfg.dirs.shared.mkdir(parents=True, exist_ok=True)
    sizes = read_json(cfg.dirs.shared / "client_data_sizes.json")
    model_info = read_json(cfg.dirs.shared / "model_size.json")
    model_bytes = int(model_info["bytes"])
    result = ({str(k): int(v) for k, v in sizes.items()}, model_bytes)
    print(f"[FUNC _read_shared_inputs] Loaded {len(result[0])} clients, model_size={model_bytes} bytes")
    return result

def _append_round_log(csv_path: Path, round_id: int, selected: List[str], times: Dict[str, Dict[str, float]], deadline_sec: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["round_id", "client_id", "Tcmp_sec", "Ttx_sec", "Ttot_sec", "round_deadline_sec"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        for cid in selected:
            t = times[cid]
            w.writerow([round_id, cid, f"{t['Tcmp']:.6f}", f"{t['Ttx']:.6f}", f"{t['Ttot']:.6f}", f"{deadline_sec:.6f}"])
    print(f"[FUNC _append_round_log] Round {round_id:03d}: wrote {len(selected)} client logs -> {csv_path}")

def simulate_client_training(model, client_id, epochs):
    print(f"  [SIMULATING] Training for client {client_id} for {epochs} epochs...")
    trained_model = copy.deepcopy(model)
    for param in trained_model.parameters():
        noise = torch.randn(param.size(), device=model.parameters().__next__().device) * 0.01 
        param.data.add_(noise)
    return trained_model.state_dict()

def run_edge(sim_config_path: str | Path, scheduler_type: str, output_dir_override: str) -> None:
    cfg = load_sim_config(sim_config_path)
    
    project_root = Path(sim_config_path).parent.parent.parent
    base_output_dir = project_root / output_dir_override
    shared_data_dir = project_root / "cfl_edge/outputs/shared"

    new_dirs = SimDirs(
        shared=shared_data_dir,
        edge_rounds=(base_output_dir / "edge/rounds"),
        edge_logs=(base_output_dir / "edge/logs"),
        ml_rounds=(base_output_dir / "ml/rounds"),
    )
    object.__setattr__(cfg, 'dirs', new_dirs)
    
    R = cfg.rounds_R

    data_sizes, model_bytes = _read_shared_inputs(cfg)
    clients = build_clients_from_sizes(data_sizes, cfg)
    all_ids: List[str] = sorted(clients.keys())
    
    optimized_hyperparams = Hyperparameters(
        width_mult=1.8574, learning_rate=0.077489, batch_size=64,
        dropout_rate=0.1426, weight_decay=0.000159, momentum=0.8017,
        conv_channels=[16, 32, 64, 128]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EDGE INIT] Using device: {device}")
    
    initial_model_path = "ci/ga_optimized_model.pth"
    initial_state_dict = torch.load(initial_model_path, map_location=device)
    
    cluster_models = { "base": initial_state_dict }
    client_to_cluster_map = {cid: "base" for cid in all_ids}
    cluster_stats = {"base": {"stationary": False, "avg_delta_norm": float('inf')}}
    client_selection_counts = {cid: 0 for cid in all_ids}

    print(f"[EDGE INIT] Loaded {len(all_ids)} clients | Rounds={R} | Model={model_bytes} bytes")
    print(f"[EDGE INIT] Running simulation with '{scheduler_type.upper()}' scheduler.")

    for r in range(1, R + 1):
        print(f"\n--- [EDGE] Starting Round {r:03d}/{R} ---")
        
        eligible_ids = [cid for cid in all_ids if clients[cid].available]
        if not eligible_ids: continue
            
        eligible_clients_map = {cid: clients[cid] for cid in eligible_ids}
        all_times, _ = estimate_times_for_selected(eligible_clients_map, model_bytes, cfg)
        for cid, t in all_times.items():
            clients[cid].Ttot = t['Ttot']
        
        if scheduler_type == 'advanced':
            selected_ids = select_clients_advanced(eligible_clients_map, cfg.max_clients_per_round, client_to_cluster_map, cluster_stats, client_selection_counts)
        else: # 'baseline'
            selected_ids = select_random(eligible_ids, cfg.max_clients_per_round)

        if not selected_ids: continue
        
        client_updates = {}
        original_models = {}
        for cid in selected_ids:
            cluster_id = client_to_cluster_map[cid]
            model_to_train = CNN(hyperparams=optimized_hyperparams).to(device)
            model_to_train.load_state_dict(cluster_models[cluster_id])
            original_models[cid] = copy.deepcopy(model_to_train.state_dict())
            updated_weights = simulate_client_training(model_to_train, cid, cfg.local_epochs_E)
            delta = {name: updated_weights[name] - original_models[cid][name] for name in updated_weights}
            client_updates[cid] = delta

        updates_by_cluster = {}
        for cid in selected_ids:
            cluster_id = client_to_cluster_map[cid]
            if cluster_id not in updates_by_cluster:
                updates_by_cluster[cluster_id] = []
            updates_by_cluster[cluster_id].append(client_updates[cid])
        
        for cluster_id, deltas in updates_by_cluster.items():
            avg_delta = { name: torch.stack([d[name] for d in deltas if name in d and d[name].dtype.is_floating_point]).mean(dim=0) for name in deltas[0] if deltas[0][name].dtype.is_floating_point }
            current_model_state = cluster_models[cluster_id]
            for name in avg_delta: current_model_state[name] += avg_delta[name]
            
            avg_delta_norm = torch.linalg.norm(torch.cat([p.flatten() for p in avg_delta.values()]))
            if cluster_id not in cluster_stats: cluster_stats[cluster_id] = {"stationary": False}
            cluster_stats[cluster_id]["avg_delta_norm"] = float(avg_delta_norm)
            
            STATIONARY_THRESHOLD = 0.1 
            if avg_delta_norm < STATIONARY_THRESHOLD:
                if not cluster_stats[cluster_id].get("stationary", False):
                    cluster_stats[cluster_id]["stationary"] = True

        delta_vectors = [torch.cat([p.flatten() for p in client_updates[cid].values() if p.dtype.is_floating_point]).cpu().numpy() for cid in selected_ids]
        distance_matrix = compute_pairwise_dis(delta_vectors, metric='cosine')
        
        max_k = len(selected_ids) - 1 if len(selected_ids) > 1 else 1
        if max_k > 1:
            best_labels, best_k, best_score = adaptive_kmeans(distance_matrix, max_clusters=max_k)
            if best_labels is not None and best_k > 1:
                print(f"[CFL] Split detected! Best k={best_k} with score={best_score:.4f}.")
                for client_idx, cluster_label in enumerate(best_labels):
                    client_id = selected_ids[client_idx]
                    parent_cluster = client_to_cluster_map[client_id]
                    new_cluster_id = f"{parent_cluster}_{cluster_label}"
                    if new_cluster_id not in cluster_models:
                        cluster_models[new_cluster_id] = copy.deepcopy(cluster_models[parent_cluster])
                        cluster_stats[new_cluster_id] = {"stationary": False, "avg_delta_norm": float('inf')}
                    client_to_cluster_map[client_id] = new_cluster_id
            else:
                print("[CFL] No split detected (k=1 was optimal).")
        else:
            print("[CFL] No split detected (not enough clients to form clusters).")

        times_for_selected_clients = {cid: all_times[cid] for cid in selected_ids}
        _, deadline_sec = estimate_times_for_selected({cid: clients[cid] for cid in selected_ids}, model_bytes, cfg)
        
        agg_sets = build_aggregation_sets(selected_ids, cfg.subchannels_N)
        plan = { "round_id": r, "selected_clients": [{"id": cid, "cluster": client_to_cluster_map[cid]} for cid in selected_ids], "aggregation_sets": agg_sets, "subchannels_N": cfg.subchannels_N, "deadline_sec": deadline_sec }
        write_json(plan, cfg.dirs.edge_rounds / f"round_{r:03d}_plan.json")
        _append_round_log(cfg.dirs.edge_logs / "round_times.csv", r, selected_ids, times_for_selected_clients, deadline_sec)

        print(f"[EDGE] Round {r:03d}: selected={len(selected_ids)} clients | deadline={deadline_sec:.3f}s")