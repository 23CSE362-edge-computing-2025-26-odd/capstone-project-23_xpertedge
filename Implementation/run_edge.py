from __future__ import annotations
import argparse
from pathlib import Path
import sys

# This line adds the 'Implementation' directory to the Python path
sys.path.append(str(Path(__file__).parent))

from cfl_edge.src.edge.orchestrator import run_edge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="cfl_edge/configs/sim_edge_params.yaml")
    args = ap.parse_args()
    run_edge(args.config)

if __name__ == "__main__":
    main()