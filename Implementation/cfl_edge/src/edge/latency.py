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
    result = allocated_bw_hz * math.log(1.0 + snr)
    print(
        f"[FUNC achievable_rate_bps] cid={c.cid}, bw={allocated_bw_hz:.2e} Hz, "
        f"noise={noise_watt:.2e} W -> rate={result:.6e} bps"
    )
    return result


# Computes transmission time (seconds) for sending a model of given size over a channel.
def tx_time_sec(model_bytes: int, rate_bps: float) -> float:
    # Time = data size / rate (avoid divide by zero with small epsilon)
    result = float(model_bytes) / max(rate_bps, 1e-30)
    print(
        f"[FUNC tx_time_sec] model_size={model_bytes} bytes, "
        f"rate={rate_bps:.6e} bps -> time={result:.6f} s"
    )
    return result


# Computes local training time (seconds) based on dataset size, CPU speed, and epochs.
def compute_time_sec(E: int, cycles_per_sample: int, data_size: int, cpu_freq_hz: float) -> float:
    total_cycles = E * cycles_per_sample * data_size   # Total cycles needed
    result = total_cycles / max(cpu_freq_hz, 1e-30)    # Time = cycles / frequency
    print(
        f"[FUNC compute_time_sec] E={E}, cycles/sample={cycles_per_sample}, "
        f"data_size={data_size}, cpu={cpu_freq_hz:.2e} Hz "
        f"-> time={result:.6f} s"
    )
    return result


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

        print(
            f"[CLIENT {cid}] Tcmp={tcmp:.6f} s, Ttx={ttx:.6f} s, Ttot={ttot:.6f} s"
        )

    print(f"[FUNC estimate_times_for_selected] -> deadline={max_deadline:.6f} s")
    return out, max_deadline
