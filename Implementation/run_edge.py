# FILE: Implementation/run_edge.py

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import json

# This adds the 'Implementation' directory to the Python path
sys.path.append(str(Path(__file__).parent))

from cfl_edge.src.edge.orchestrator import run_edge

def evaluate_and_print_metrics(advanced_dir: Path, baseline_dir: Path):
    """
    Loads results from two simulation runs, calculates key edge metrics,
    and prints a formatted comparison table.
    """
    print("\n" + "="*80)
    print(" " * 25 + "EVALUATION METRICS COMPARISON")
    print("="*80)

    try:
        # --- Load Data ---
        df_adv = pd.read_csv(advanced_dir / "edge/logs/round_times.csv")
        df_base = pd.read_csv(baseline_dir / "edge/logs/round_times.csv")

        # --- Metric 1: Average Round Deadline (Latency) ---
        avg_deadline_adv = df_adv['round_deadline_sec'].mean()
        avg_deadline_base = df_base['round_deadline_sec'].mean()

        # --- Metric 2: Total Training Time ---
        total_time_adv = df_adv['round_deadline_sec'].sum()
        total_time_base = df_base['round_deadline_sec'].sum()
        # --- Metric 3: Fairness (Std. Dev. of Selections) ---
        fairness_adv = df_adv['client_id'].value_counts().std()
        fairness_base = df_base['client_id'].value_counts().std()

        # --- Metric 4: Convergence Speed (Rounds to First Split) ---
        def find_first_split(run_dir: Path):
            for round_file in sorted(run_dir.glob("edge/rounds/round_*_plan.json")):
                with open(round_file, 'r') as f:
                    plan = json.load(f)
                    clusters = {c['cluster'] for c in plan['selected_clients']}
                    if len(clusters) > 1:
                        return plan['round_id']
            return "N/A"

        split_round_adv = find_first_split(advanced_dir)
        split_round_base = find_first_split(baseline_dir)

        # --- Print Formatted Table ---
        print(f"{'Metric':<40} | {'Advanced Scheduler':<25} | {'Baseline Scheduler':<25}")
        print("-" * 80)
        print(f"{'Avg. Round Deadline (s) [Lower is Better]':<40} | {avg_deadline_adv:<25.2f} | {avg_deadline_base:<25.2f}")
        print(f"{'Total Training Time (s) [Lower is Better]':<40} | {total_time_adv:<25.2f} | {total_time_base:<25.2f}")
        print(f"{'Fairness (Std Dev) [Lower is Better]':<40} | {fairness_adv:<25.2f} | {fairness_base:<25.2f}")
        print(f"{'Rounds to First Split [Lower is Better]':<40} | {str(split_round_adv):<25} | {str(split_round_base):<25}")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n[EVALUATION ERROR] Could not find result files. Make sure both simulations ran correctly.")
        print(f"Missing file: {e.filename}")
    except Exception as e:
        print(f"\n[EVALUATION ERROR] An error occurred during evaluation: {e}")


def main():
    ap = argparse.ArgumentParser(description="Run and Evaluate the CFL Edge Simulation.")
    ap.add_argument("--config", type=str, default="cfl_edge/configs/sim_edge_params.yaml")
    args = ap.parse_args()

    # Define output directories
    advanced_output = "cfl_edge/outputs"
    baseline_output = "cfl_edge/outputs_baseline"

    # --- Run 1: Advanced Scheduler ---
    print("\n" + "#"*80)
    print(" " * 25 + "RUNNING: ADVANCED SCHEDULER")
    print("#"*80)
    run_edge(args.config, scheduler_type='advanced', output_dir_override=advanced_output)

    # --- Run 2: Baseline Scheduler ---
    print("\n" + "#"*80)
    print(" " * 25 + "RUNNING: BASELINE SCHEDULER")
    print("#"*80)
    run_edge(args.config, scheduler_type='baseline', output_dir_override=baseline_output)

    # --- Step 3: Evaluate and Print Metrics ---
    evaluate_and_print_metrics(Path(advanced_output), Path(baseline_output))


if __name__ == "__main__":
    main()