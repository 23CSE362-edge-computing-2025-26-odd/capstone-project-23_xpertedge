from __future__ import annotations
import argparse
from pathlib import Path
from src.edge.orchestrator import run_edge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/sim_edge_params.yaml")
    args = ap.parse_args()
    run_edge(args.config)

if __name__ == "__main__":
    main()
