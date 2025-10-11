from dataclasses import dataclass
from typing import Dict
import random

@dataclass
class ClientProfile:
    # Represents a client with dataset size, distance, channel gain, CPU frequency, 
    # transmission power, and availability status.
    cid: str
    data_size: int
    distance_m: float
    channel_gain_lin: float
    cpu_freq_hz: float
    cycles_per_sample: int
    tx_power_watt: float
    available: bool = True


# Converts power from dBm to Watts.
def _dbm_to_watt(dbm: float) -> float:
    result = 10 ** ((dbm - 30) / 10)
    print(f"[FUNC _dbm_to_watt] Input: {dbm:.2f} dBm -> Output: {result:.6e} W")
    return result


# Converts decibel (dB) values to linear scale.
def _db_to_lin(db: float) -> float:
    result = 10 ** (db / 10)
    print(f"[FUNC _db_to_lin] Input: {db:.2f} dB -> Output: {result:.6e} (linear)")
    return result


# Calculates wireless channel gain using the pathloss model (linear scale).
def _pathloss_gain_linear(g0_dB: float, d0: float, d: float, alpha: float) -> float:
    g0_lin = _db_to_lin(g0_dB)               # Convert reference gain from dB to linear
    result = g0_lin * (d0 / d) ** alpha      # Apply pathloss formula
    print(
        f"[FUNC _pathloss_gain_linear] g0_dB={g0_dB:.2f}, d0={d0:.2f}, d={d:.2f}, alpha={alpha:.2f} "
        f"-> Output: {result:.6e}"
    )
    return result


# Builds a dictionary of client profiles with random distance, CPU frequency, 
# transmit power, and channel gain based on simulation configuration.
# Returns all clients as ClientProfile objects mapped by their IDs.
def build_clients_from_sizes(data_sizes: Dict[str, int], cfg: 'SimConfig') -> Dict[str, ClientProfile]:
    clients: Dict[str, ClientProfile] = {}
    for cid, sz in data_sizes.items():
        d_min, d_max = cfg.distance_m_range
        distance_m = random.uniform(d_min, d_max)

        gain_lin = _pathloss_gain_linear(
            cfg.pathloss_g0_dB, cfg.ref_distance_m, distance_m, cfg.pathloss_exponent
        )

        f_min, f_max = cfg.cpu_freq_GHz_range
        cpu_freq_hz = random.uniform(f_min, f_max) * 1e9

        pmin, pmax = cfg.tx_power_dBm_range
        tx_power_watt = _dbm_to_watt(random.uniform(pmin, pmax))

        clients[cid] = ClientProfile(
            cid=cid,
            data_size=int(sz),
            distance_m=distance_m,
            channel_gain_lin=gain_lin,
            cpu_freq_hz=cpu_freq_hz,
            cycles_per_sample=cfg.cycles_per_sample,
            tx_power_watt=tx_power_watt,
            available=True,
        )

        # Print each client profile in a clean format
        print(
            f"[Client {cid}] size={sz}, dist={distance_m:.2f} m, gain={gain_lin:.6e}, "
            f"cpu={cpu_freq_hz/1e9:.2f} GHz, tx_power={tx_power_watt:.6e} W"
        )

    print(f"[FUNC build_clients_from_sizes] -> Built {len(clients)} clients")
    return clients
