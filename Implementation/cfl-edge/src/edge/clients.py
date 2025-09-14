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
    return 10 ** ((dbm - 30) / 10)


# Converts decibel (dB) values to linear scale.
def _db_to_lin(db: float) -> float:
    return 10 ** (db / 10)


# Calculates wireless channel gain using the pathloss model (linear scale).
def _pathloss_gain_linear(g0_dB: float, d0: float, d: float, alpha: float) -> float:
    g0_lin = _db_to_lin(g0_dB)               # Convert reference gain from dB to linear
    return g0_lin * (d0 / d) ** alpha        # Apply pathloss formula: decreases with distance and environment factor


# Builds a dictionary of client profiles with random distance, CPU frequency, 
# transmit power, and channel gain based on simulation configuration.
# Returns all clients as ClientProfile objects mapped by their IDs.
def build_clients_from_sizes(data_sizes: Dict[str, int], cfg: SimConfig) -> Dict[str, ClientProfile]:
    clients: Dict[str, ClientProfile] = {}
    for cid, sz in data_sizes.items():
        # Pick a random distance for the client within the configured range
        # (simulates clients being at different locations from the server).
        d_min, d_max = cfg.distance_m_range
        distance_m = random.uniform(d_min, d_max)

        # Compute channel gain using pathloss model
        # (important to model wireless communication quality).
        gain_lin = _pathloss_gain_linear(
            cfg.pathloss_g0_dB, cfg.ref_distance_m, distance_m, cfg.pathloss_exponent
        )

        # Pick a random CPU frequency within range (converted GHz → Hz),
        # simulating heterogeneous client devices.
        f_min, f_max = cfg.cpu_freq_GHz_range
        cpu_freq_hz = random.uniform(f_min, f_max) * 1e9

        # Pick a random transmission power (converted dBm → Watt),
        # to simulate varying power capabilities of devices.
        pmin, pmax = cfg.tx_power_dBm_range
        tx_power_watt = _dbm_to_watt(random.uniform(pmin, pmax))

        # Create a ClientProfile object with all parameters
        # and store it in the dictionary keyed by client ID.
        clients[cid] = ClientProfile(
            cid=cid,
            data_size=int(sz),              # Dataset size for this client
            distance_m=distance_m,          # Distance from server
            channel_gain_lin=gain_lin,      # Computed wireless channel gain
            cpu_freq_hz=cpu_freq_hz,        # CPU frequency in Hz
            cycles_per_sample=cfg.cycles_per_sample,  # Training complexity factor
            tx_power_watt=tx_power_watt,    # Transmission power in Watts
            available=True,                 # Client is available to participate
        )
    return clients
