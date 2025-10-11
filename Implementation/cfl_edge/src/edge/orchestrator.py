

from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import csv
import time
import torch
import copy

# --- CORRECTED IMPORTS for CI + Edge ---
from cfl_edge.src.edge.configs import load_sim_config, SimConfig
from cfl_edge.src.edge.clients import build_clients_from_sizes, ClientProfile
from cfl_edge.src.edge.latency import estimate_times_for_selected
from cfl_edge.src.utils.io_json import read_json, write_json
from ci.src.models import CNN
from ci.src.genetic import Hyperparameters

# Using the baseline random scheduler, not the advanced one
from cfl_edge.src.edge.scheduler_baseline import select_random, build_aggregation_sets

def _read_shared_inputs(cfg: SimConfig) -> tuple[Dict[str, int], int]:
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
        if write_header:
            w.writerow(header)
        for cid in selected:
            t = times[cid]
            w.writerow([round_id, cid, f"{t['Tcmp']:.6f}", f"{t['Ttx']:.6f}", f"{t['Ttot']:.6f}", f"{deadline_sec:.6f}"])
    print(f"[FUNC _append_round_log] Round {round_id:03d}: wrote {len(selected)} client logs -> {csv_path}")

def simulate_client_training(model, client_id, epochs):
    print(f"  [SIMULATING] Training for client {client_id} for {epochs} epochs...")
    trained_model = copy.deepcopy(model)
    for param in trained_model.parameters():
        noise = torch.randn(param.size(), device=model.parameters()._next_().device) * 0.01 
        param.data.add_(noise)
    return trained_model.state_dict()

def run_edge(sim_config_path: str | Path, rounds_override: int | None = None, dry_run_override: bool | None = None) -> None:
    cfg = load_sim_config(sim_config_path)
    R = rounds_override if rounds_override is not None else cfg.rounds_R
    if dry_run_override is True:
        object._setattr_(cfg, "block_on_ml_updates", False)

    data_sizes, model_bytes = _read_shared_inputs(cfg)
    clients = build_clients_from_sizes(data_sizes, cfg)
    all_ids: List[str] = sorted(clients.keys())
    
    # Define the exact hyperparameters of your saved model
    optimized_hyperparams = Hyperparameters(
        width_mult=1.8574, learning_rate=0.077489, batch_size=64,
        dropout_rate=0.1426, weight_decay=0.000159, momentum=0.8017,
        conv_channels=[16, 32, 64, 128]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EDGE INIT] Using device: {device}")
    
    # Load the initial global model from your 'ci' folder
    initial_model_path = "ci/ga_optimized_model.pth"
    global_model_state = torch.load(initial_model_path, map_location=device)
    
    print(f"[EDGE INIT] Loaded {len(all_ids)} clients | Rounds={R} | Model={model_bytes} bytes")

    # --- MAIN FEDERATED LEARNING LOOP (Standard FedAvg) ---
    for r in range(1, R + 1):
        print(f"\n--- [EDGE] Starting Round {r:03d}/{R} ---")
        
        eligible_ids = [cid for cid in all_ids if clients[cid].available]
        if not eligible_ids:
            print(f"[EDGE] Round {r:03d}: No eligible clients. Stopping simulation.")
            break
            
        # --- 1. CLIENT SELECTION (Using baseline random scheduler) ---
        selected_ids = select_random(eligible_ids, cfg.max_clients_per_round)

        if not selected_ids:
            print(f"[EDGE] Round {r:03d}: No clients selected. Skipping round.")
            continue
        
        # --- 2. SIMULATE CLIENT TRAINING ---
        client_deltas = []
        for cid in selected_ids:
            # Create a model instance with the correct architecture
            model_to_train = CNN(hyperparams=optimized_hyperparams).to(device)
            model_to_train.load_state_dict(global_model_state)
            
            original_weights = copy.deepcopy(model_to_train.state_dict())
            updated_weights = simulate_client_training(model_to_train, cid, cfg.local_epochs_E)
            
            # Calculate the delta (the client's update)
            delta = {name: updated_weights[name] - original_weights[name] for name in updated_weights}
            client_deltas.append(delta)

        # --- 3. FEDERATED AVERAGING (FedAvg) ---
        print(f"[EDGE] Aggregating updates for the global model...")
        
        # Average the deltas from all clients
        avg_delta = {
            name: torch.stack([d[name] for d in client_deltas if name in d and d[name].dtype.is_floating_point]).mean(dim=0)
            for name in client_deltas[0] if client_deltas[0][name].dtype.is_floating_point
        }
        
        # Apply the averaged update to the global model
        for name in avg_delta:
            global_model_state[name] += avg_delta[name]

        # --- 4. LOGGING ---
        selected_map = {cid: clients[cid] for cid in selected_ids}
        times, deadline_sec = estimate_times_for_selected(selected_map, model_bytes, cfg)
        
        agg_sets = build_aggregation_sets(selected_ids, cfg.subchannels_N)
        plan = {
            "round_id": r,
            # In this simple version, all clients belong to the 'base' cluster
            "selected_clients": [{"id": cid, "cluster": "base"} for cid in selected_ids],
            "aggregation_sets": agg_sets,
            "subchannels_N": cfg.subchannels_N,
            "deadline_sec": deadline_sec,
        }
        write_json(plan, cfg.dirs.edge_rounds / f"round_{r:03d}_plan.json")
        _append_round_log(cfg.dirs.edge_logs / "round_times.csv", r, selected_ids, times, deadline_sec)

        print(f"[EDGE] Round {r:03d}: selected={len(selected_ids)} clients | deadline={deadline_sec:.3f}s")