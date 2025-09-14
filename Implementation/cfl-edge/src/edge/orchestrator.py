from _future_ import annotations # allows us to use classes/functions in type hints.
from typing import Dict, List # helpers - for hints.
from pathlib import Path
import csv  # importing modules 
import time

from .configs import load_sim_config, SimConfig # import simulation configuration tools
from .clients import build_clients_from_sizes, ClientProfile # implement client utility
from .latency import estimate_times_for_selected # import latency estimation utility
from .scheduler_baseline import select_random, build_aggregation_sets # baseline scheduling utility
from ..utils.io_json import read_json, write_json # json input or output utility

def _read_shared_inputs(cfg: SimConfig) -> tuple[Dict[str, int], int]:
    sizes = read_json(cfg.dirs.shared / "client_data_sizes.json")  # cid - int -- read from json file
    model_info = read_json(cfg.dirs.shared / "model_size.json")    # bytes - int -- read from json file
    model_bytes = int(model_info["bytes"])
    return {str(k): int(v) for k, v in sizes.items()}, model_bytes

# append all the rounds of the computation time and transmission time and all
def _append_round_log(csv_path: Path, round_id: int, selected: List[str], times: Dict[str, Dict[str, float]], deadline_sec: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    #Define the header with column names
    header = ["round_id", "client_id", "Tcmp_sec", "Ttx_sec", "Ttot_sec", "round_deadline_sec"]
    write_header = not csv_path.exists()
    #opens the csv file in append mode
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:  
            w.writerow(header) # writes header only for the first time creating a file
        for cid in selected:  # for every selected client log their timing info - transmission time, computation time.
            t = times[cid]
            w.writerow([round_id, cid, f"{t['Tcmp']:.6f}", f"{t['Ttx']:.6f}", f"{t['Ttot']:.6f}", f"{deadline_sec:.6f}"])

def _wait_for_ml_updates(cfg: SimConfig, round_id: int) -> None:
    if not cfg.block_on_ml_updates:
        return
      # Construct the expected JSON file path for this round's ML updates
    target = cfg.dirs.ml_rounds / f"round_{round_id:03d}_updates.json"
    t0 = time.time()

    # Keep polling until the updates file appears or timeout occurs
    while True:
         #  If the update file exists, stop waiting and return
        if target.exists():
            return
        #If the wait time exceeds the allowed timeout, raise an error
        if time.time() - t0 > cfg.ml_updates_timeout_sec:
            raise TimeoutError(f"Timed out waiting for ML updates: {target}")
        time.sleep(cfg.poll_interval_sec)

def run_edge(sim_config_path: str | Path, rounds_override: int | None = None, dry_run_override: bool | None = None) -> None:
    cfg = load_sim_config(sim_config_path)  # Loads the simulation configuration files

    # Determines the no of rounds
    R = rounds_override if rounds_override is not None else cfg.rounds_R
    if dry_run_override is True:
        # SimConfig is frozen; need to adjust at runtime
        object._setattr_(cfg, "block_on_ml_updates", False)  # type: ignore

    data_sizes, model_bytes = _read_shared_inputs(cfg) # data sizes and model size
    clients = build_clients_from_sizes(data_sizes, cfg) # these are build from client sizes
    all_ids: List[str] = sorted(clients.keys()) # sort all the ids of the clients

    for r in range(1, R + 1):
        # Eligible = checks for the eligible clients and chooses the available clients
        eligible_ids = [cid for cid in all_ids if clients[cid].available]

        # Random selection
        selected_ids = select_random(eligible_ids, cfg.max_clients_per_round)
        agg_sets = build_aggregation_sets(selected_ids, cfg.subchannels_N)

        # Compute times/deadline for selected clients
        selected_map: Dict[str, ClientProfile] = {cid: clients[cid] for cid in selected_ids}
        times, deadline_sec = estimate_times_for_selected(selected_map, model_bytes, cfg)

        # Write plan JSON (cluster='base' for now)
        plan = {
            "round_id": r,
            "selected_clients": [{"id": cid, "cluster": "base"} for cid in selected_ids],
            "aggregation_sets": agg_sets,
            "subchannels_N": cfg.subchannels_N,
            "deadline_sec": deadline_sec,
        }
        write_json(plan, cfg.dirs.edge_rounds / f"round_{r:03d}_plan.json")

        # Append timing log for each client
        _append_round_log(cfg.dirs.edge_logs / "round_times.csv", r, selected_ids, times, deadline_sec)

        # Console summary
        print(f"[EDGE] Round {r:03d}: selected={selected_ids} deadline={deadline_sec:.3f}s groups={agg_sets}")

        # Wait for ML to finish this round 
        _wait_for_ml_updates(cfg, r)
