from _future_ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any
import yaml
import random
import numpy as np

@dataclass(frozen=True)
class SimDirs:
    shared: Path
    edge_rounds: Path
    edge_logs: Path
    ml_rounds: Path

@dataclass(frozen=True)
class SimConfig:
    seed: int
    bandwidth_MHz: float
    subchannels_N: int
    noise_watt: float

    distance_m_range: Tuple[float, float]
    pathloss_g0_dB: float
    ref_distance_m: float
    pathloss_exponent: float

    cpu_freq_GHz_range: Tuple[float, float]
    cycles_per_sample: int
    local_epochs_E: int

    tx_power_dBm_range: Tuple[float, float]

    rounds_R: int
    max_clients_per_round: int

    block_on_ml_updates: bool
    ml_updates_timeout_sec: int
    poll_interval_sec: float

    dirs: SimDirs

def _as_tuple2(v) -> Tuple[float, float]:
    assert isinstance(v, (list, tuple)) and len(v) == 2
    return (float(v[0]), float(v[1]))

def load_sim_config(path: str | Path) -> SimConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    dirs = SimDirs(
        shared=Path(cfg["dirs"]["shared"]),
        edge_rounds=Path(cfg["dirs"]["edge_rounds"]),
        edge_logs=Path(cfg["dirs"]["edge_logs"]),
        ml_rounds=Path(cfg["dirs"]["ml_rounds"]),
    )

    sc = SimConfig(
        seed=int(cfg["seed"]),
        bandwidth_MHz=float(cfg["bandwidth_MHz"]),
        subchannels_N=int(cfg["subchannels_N"]),
        noise_watt=float(cfg["noise_watt"]),

        distance_m_range=_as_tuple2(cfg["distance_m_range"]),
        pathloss_g0_dB=float(cfg["pathloss_g0_dB"]),
        ref_distance_m=float(cfg["ref_distance_m"]),
        pathloss_exponent=float(cfg["pathloss_exponent"]),

        cpu_freq_GHz_range=_as_tuple2(cfg["cpu_freq_GHz_range"]),
        cycles_per_sample=int(cfg["cycles_per_sample"]),
        local_epochs_E=int(cfg["local_epochs_E"]),

        tx_power_dBm_range=_as_tuple2(cfg["tx_power_dBm_range"]),

        rounds_R=int(cfg["rounds_R"]),
        max_clients_per_round=int(cfg["max_clients_per_round"]),

        block_on_ml_updates=bool(cfg["block_on_ml_updates"]),
        ml_updates_timeout_sec=int(cfg["ml_updates_timeout_sec"]),
        poll_interval_sec=float(cfg["poll_interval_sec"]),

        dirs=dirs,
    )

    # seeds
    random.seed(sc.seed)
    np.random.seed(sc.seed)

    # make sure dirs exist
    sc.dirs.edge_rounds.mkdir(parents=True, exist_ok=True)
    sc.dirs.edge_logs.mkdir(parents=True, exist_ok=True)

    return sc