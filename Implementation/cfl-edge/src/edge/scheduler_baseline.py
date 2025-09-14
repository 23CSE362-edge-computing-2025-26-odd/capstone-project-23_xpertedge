from __future__ import annotations
from typing import List
import random
import math

def select_random(eligible_ids: List[str], max_clients_per_round: int) -> List[str]:
    if max_clients_per_round <= 0:
        result: List[str] = []
        print(f"select_random -> {result}")
        return result
    n = min(len(eligible_ids), max_clients_per_round)
    result = random.sample(eligible_ids, n)
    print(f"select_random -> {result}")
    return result

def build_aggregation_sets(selected_ids: List[str], subchannels_N: int) -> List[List[str]]:
    """
    Split selected clients into groups of size ≤ N, to upload in batches (OFDMA sets).
    """
    if subchannels_N <= 0:
        result = [selected_ids] if selected_ids else []
        print(f"build_aggregation_sets -> {result}")
        return result
    groups: List[List[str]] = []
    for i in range(0, len(selected_ids), subchannels_N):
        groups.append(selected_ids[i : i + subchannels_N])
    print(f"build_aggregation_sets -> {groups}")
    return groups
