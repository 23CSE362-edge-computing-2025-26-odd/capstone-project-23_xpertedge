from __future__ import annotations
from typing import Dict, List, Tuple
import math

from .clients import ClientProfile
from .configs import SimConfig


# Calculates the achievable data rate (bps) for a client using Shannon’s formula.
def achievable_rate_bps(c: ClientProfile, allocated_bw_hz: float, noise_watt: float) -> float:
    # Signal-to-noise ratio (SNR) = (Transmit power × Channel gain) / Noise power
    snr = (c.tx_power_watt * c.channel_gain_lin) / max(noise_watt, 1e-30)
    # Achievable rate = Bandwidth × log(1 + SNR)
    return allocated_bw_hz * math.log(1.0 + snr)


# Computes transmission time (seconds) for sending a model of given size over a channel.
def tx_time_sec(model_bytes: int, rate_bps: float) -> float:
    # Time = data size / rate (avoid divide by zero with small epsilon)
    return float(model_bytes) / max(rate_bps, 1e-30)


# Computes local training time (seconds) based on dataset size, CPU speed, and epochs.
def compute_time_sec(E: int, cycles_per_sample: int, data_size: int, cpu_freq_hz: float) -> float:
    total_cycles = E * cycles_per_sample * data_size   # Total cycles needed
    return total_cycles / max(cpu_freq_hz, 1e-30)      # Time = cycles / frequency


# Estimates computation, transmission, and total times for selected clients.
def estimate_times_for_selected(
    selected: Dict[str, ClientProfile],
    model_bytes: int,
    cfg: SimConfig,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    Returns:
      times: {cid: {"Tcmp":..., "Ttx":..., "Ttot":...}}
      deadline_sec: maximum total time across clients
    """
    B_hz = cfg.bandwidth_MHz * 1e6          # Convert bandwidth from MHz to Hz
    N = max(cfg.subchannels_N, 1)           # Number of subchannels (at least 1 to avoid /0)
    bw_alloc = B_hz / N                     # Each client gets equal share of bandwidth (OFDMA)

    out: Dict[str, Dict[str, float]] = {}   # Stores times for each client
    max_deadline = 0.0                      # Track the maximum time (slowest client)

    for cid, c in selected.items():
        r_bps = achievable_rate_bps(c, bw_alloc, cfg.noise_watt)  # Compute achievable rate
        ttx = tx_time_sec(model_bytes, r_bps)                     # Transmission time
        tcmp = compute_time_sec(cfg.local_epochs_E, c.cycles_per_sample, c.data_size, c.cpu_freq_hz)  # Computation time
        ttot = ttx + tcmp                                         # Total time = compute + transmit

        out[cid] = {"Tcmp": tcmp, "Ttx": ttx, "Ttot": ttot}       # Store results
        if ttot > max_deadline:                                   # Update if this client is the slowest
            max_deadline = ttot

    return out, max_deadline
