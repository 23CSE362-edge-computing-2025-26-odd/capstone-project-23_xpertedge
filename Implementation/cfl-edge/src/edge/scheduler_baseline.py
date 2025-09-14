from __future__ import annotations
from typing import List
import random
import math

def select_random(eligible_ids: List[str], max_clients_per_round: int) -> List[str]:
    if max_clients_per_round <= 0:
        return []
    n = min(len(eligible_ids), max_clients_per_round)
    return random.sample(eligible_ids, n)

def build_aggregation_sets(selected_ids: List[str], subchannels_N: int) -> List[List[str]]:
    """
    Split selected clients into groups of size ≤ N, to upload in batches (OFDMA sets).
    """
    if subchannels_N <= 0:
        return [selected_ids] if selected_ids else []
    groups: List[List[str]] = []
    for i in range(0, len(selected_ids), subchannels_N):
        groups.append(selected_ids[i : i + subchannels_N])
    return groups
