# FILE: Implementation/visualize_results.py

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
ADVANCED_OUTPUT_DIR = Path("cfl_edge/outputs")
BASELINE_OUTPUT_DIR = Path("cfl_edge/outputs_baseline")


def evaluate_and_print_metrics(advanced_dir: Path, baseline_dir: Path):
    """Loads results, calculates key edge metrics, and prints all values for comparison."""
    print("\n" + "="*100)
    print(" " * 30 + "DETAILED EVALUATION METRICS")
    print("="*100)

    try:
        # --- Load data ---
        df_adv = pd.read_csv(advanced_dir / "edge/logs/round_times.csv")
        df_base = pd.read_csv(baseline_dir / "edge/logs/round_times.csv")

        print("\n[ADVANCED] Data loaded successfully:")
        print(df_adv.head(), "\n")
        print("[BASELINE] Data loaded successfully:")
        print(df_base.head(), "\n")

        # --- Metric 1: LATENCY ---
        avg_deadline_adv = df_adv['round_deadline_sec'].mean()
        avg_deadline_base = df_base['round_deadline_sec'].mean()

        total_time_adv = df_adv.groupby('round_id')['round_deadline_sec'].first().sum()
        total_time_base = df_base.groupby('round_id')['round_deadline_sec'].first().sum()

        print("\n🕒 Round Deadlines (Advanced):")
        print(df_adv.groupby('round_id')['round_deadline_sec'].first().to_string())
        print("\n🕒 Round Deadlines (Baseline):")
        print(df_base.groupby('round_id')['round_deadline_sec'].first().to_string())

        # --- Metric 2: FAIRNESS ---
        fairness_adv = df_adv['client_id'].value_counts().std()
        fairness_base = df_base['client_id'].value_counts().std()

        print("\n👥 Client Selection Counts (Advanced):")
        print(df_adv['client_id'].value_counts().sort_index().to_string())
        print("\n👥 Client Selection Counts (Baseline):")
        print(df_base['client_id'].value_counts().sort_index().to_string())

        # --- Metric 3: CONVERGENCE SPEED ---
        def find_first_split(run_dir: Path) -> str:
            for round_file in sorted(run_dir.glob("edge/rounds/round_*_plan.json")):
                with open(round_file, 'r') as f:
                    plan = json.load(f)
                    clusters = {c['cluster'] for c in plan['selected_clients']}
                    if len(clusters) > 1:
                        print(f"[{run_dir}] Split detected at Round {plan['round_id']} | Clusters: {clusters}")
                        return str(plan['round_id'])
            print(f"[{run_dir}] No cluster split detected.")
            return "N/A (No Split)"

        split_round_adv = find_first_split(advanced_dir)
        split_round_base = find_first_split(baseline_dir)

        # --- Print detailed metrics ---
        print("\n" + "-"*100)
        print(f"{'METRIC':<45} | {'ADVANCED SCHEDULER':<25} | {'BASELINE SCHEDULER':<25}")
        print("-"*100)
        print(f"{'Average Round Deadline (s)':<45} | {avg_deadline_adv:<25.4f} | {avg_deadline_base:<25.4f}")
        print(f"{'Total Training Time (s)':<45} | {total_time_adv:<25.4f} | {total_time_base:<25.4f}")
        print(f"{'Fairness (Std Dev)':<45} | {fairness_adv:<25.4f} | {fairness_base:<25.4f}")
        print(f"{'Round of First Split':<45} | {split_round_adv:<25} | {split_round_base:<25}")
        print("="*100)

    except FileNotFoundError as e:
        print(f"\n[EVALUATION ERROR] Missing file: {e.filename}")
    except Exception as e:
        print(f"\n[EVALUATION ERROR] An error occurred during evaluation: {e}")


def load_and_process_results(output_dir: Path):
    """Loads and processes results from a single simulation run with detailed printing."""
    print(f"\n--> Processing results from: {output_dir}")
    times_csv_path = output_dir / "edge/logs/round_times.csv"

    if not times_csv_path.exists():
        print(f"    [ERROR] Missing file: {times_csv_path}")
        return None

    df_times = pd.read_csv(times_csv_path)
    rounds_dir = output_dir / "edge/rounds"
    cluster_counts = {}

    round_files = sorted(rounds_dir.glob("round_*_plan.json"))
    if not round_files:
        print(f"    [WARNING] No round plan JSON files found in {rounds_dir}")
        df_clusters = pd.Series(dtype=int)
    else:
        for round_file in round_files:
            with open(round_file, 'r') as f:
                plan = json.load(f)
                round_id = plan['round_id']
                clusters = {c['cluster'] for c in plan['selected_clients']}
                cluster_counts[round_id] = len(clusters)
                print(f"    Round {round_id:03d}: {len(clusters)} unique clusters -> {clusters}")
        df_clusters = pd.Series(cluster_counts).sort_index()

    client_selection_counts = df_times['client_id'].value_counts()

    print(f"    [INFO] Processed {len(df_times)} log entries.")
    print("    Cluster counts by round:")
    print(df_clusters.to_string())

    print("    Client selection counts:")
    print(client_selection_counts.to_string())

    return {
        "times": df_times,
        "clusters": df_clusters,
        "fairness": client_selection_counts
    }


def plot_deadline_comparison(advanced_data, baseline_data):
    """Plot 1: Deadline comparison."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    adv_deadlines = advanced_data['times'].groupby('round_id')['round_deadline_sec'].first()
    bas_deadlines = baseline_data['times'].groupby('round_id')['round_deadline_sec'].first()

    ax.plot(adv_deadlines.index, adv_deadlines.values, marker='o', label='Advanced')
    ax.plot(bas_deadlines.index, bas_deadlines.values, marker='x', linestyle='--', label='Baseline')

    ax.set_title('Round Deadline Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Round Number')
    ax.set_ylabel('Deadline (seconds)')
    ax.legend()
    plt.savefig('deadline_comparison.png')
    print("✅ Saved: deadline_comparison.png")
    plt.close()


def plot_fairness_comparison(advanced_data, baseline_data):
    """Plot 2: Fairness comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    adv_fairness = advanced_data['fairness']
    bas_fairness = baseline_data['fairness']

    ax1.bar(adv_fairness.index, adv_fairness.values, color='cornflowerblue')
    ax1.set_title(f'Advanced Fairness (Std: {adv_fairness.std():.2f})')
    ax2.bar(bas_fairness.index, bas_fairness.values, color='salmon')
    ax2.set_title(f'Baseline Fairness (Std: {bas_fairness.std():.2f})')
    plt.savefig('fairness_comparison.png')
    print("✅ Saved: fairness_comparison.png")
    plt.close()


def plot_clustering_speed(advanced_data, baseline_data):
    """Plot 3: Cluster formation."""
    fig, ax = plt.subplots(figsize=(12, 7))
    adv_clusters_growth = advanced_data['clusters'].cummax()
    bas_clusters_growth = baseline_data['clusters'].cummax()

    ax.step(adv_clusters_growth.index, adv_clusters_growth.values, label='Advanced')
    ax.step(bas_clusters_growth.index, bas_clusters_growth.values, linestyle='--', label='Baseline')
    ax.set_title('Cluster Formation Speed')
    ax.set_xlabel('Round Number')
    ax.set_ylabel('Unique Clusters')
    ax.legend()
    plt.savefig('clustering_speed.png')
    print("✅ Saved: clustering_speed.png")
    plt.close()


def main():
    """Main: print all values and generate plots."""
    print("\n--- STARTING FULL EVALUATION ---")
    adv_results = load_and_process_results(ADVANCED_OUTPUT_DIR)
    base_results = load_and_process_results(BASELINE_OUTPUT_DIR)

    if adv_results and base_results:
        evaluate_and_print_metrics(ADVANCED_OUTPUT_DIR, BASELINE_OUTPUT_DIR)
        plot_deadline_comparison(adv_results, base_results)
        plot_fairness_comparison(adv_results, base_results)
        plot_clustering_speed(adv_results, base_results)
        print("\n🎯 All metrics and plots generated successfully.")
    else:
        print("\n❌ Evaluation failed: missing data folders or log files.")


if __name__ == '__main__':
    main()
