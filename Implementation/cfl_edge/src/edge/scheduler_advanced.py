# FILE: src/edge/scheduler_advanced.py

from _future_ import annotations
from typing import Dict, List
import random
from .clients import ClientProfile

def select_clients_advanced(
    eligible_clients: Dict[str, ClientProfile],
    max_clients_per_round: int,
    client_to_cluster_map: Dict[str, str],
    cluster_stats: Dict[str, Dict],
    client_selection_counts: Dict[str, int],
) -> List[str]:
    """
    An advanced client scheduler that is latency-aware, fairness-aware, and cluster-aware.
    """
    # 1. Separate clients by their cluster
    clients_by_cluster: Dict[str, List[str]] = {}
    for cid in eligible_clients:
        cluster_id = client_to_cluster_map[cid]
        if cluster_id not in clients_by_cluster:
            clients_by_cluster[cluster_id] = []
        clients_by_cluster[cluster_id].append(cid)

    selected_clients: List[str] = []
    
    # 2. Handle stationary clusters with GREEDY scheduling first
    for cluster_id, clients_in_cluster in clients_by_cluster.items():
        if cluster_stats.get(cluster_id, {}).get("stationary", False):
            print(f"[SCHEDULER] Cluster '{cluster_id}' is stationary. Using GREEDY selection.")
            
            # Sort clients in this cluster strictly by lowest total latency
            clients_in_cluster.sort(key=lambda cid: eligible_clients[cid].Ttot)
            
            # Greedily select the fastest clients from this cluster
            num_to_select = min(len(clients_in_cluster), max_clients_per_round - len(selected_clients))
            greedy_selection = clients_in_cluster[:num_to_select]
            selected_clients.extend(greedy_selection)
            
            # Remove these clients from consideration for the general pool
            for cid in greedy_selection:
                clients_by_cluster[cluster_id].remove(cid)

    # 3. Handle non-stationary clusters with FAIRNESS-AWARE scheduling
    # Combine all remaining clients into a single pool
    remaining_pool = []
    for clients_in_cluster in clients_by_cluster.values():
        remaining_pool.extend(clients_in_cluster)
        
    # Sort remaining clients by latency to establish a baseline
    remaining_pool.sort(key=lambda cid: eligible_clients[cid].Ttot)

    # Fairness-Aware Selection: To ensure fairness, we don't just pick the fastest.
    # We select from a slightly larger pool of candidates to give slower clients a chance.
    num_to_select = min(len(remaining_pool), max_clients_per_round - len(selected_clients))
    
    # Define a candidate pool (e.g., the fastest 50%, but at least num_to_select)
    candidate_pool_size = max(num_to_select, int(len(remaining_pool) * 0.5))
    candidate_pool = remaining_pool[:candidate_pool_size]

    if num_to_select > 0:
        print(f"[SCHEDULER] Using FAIRNESS-AWARE selection for non-stationary clusters.")
        # We can introduce fairness by prioritizing those selected less often within the candidate pool
        # A simple way is to sort candidates by their selection count
        candidate_pool.sort(key=lambda cid: client_selection_counts.get(cid, 0))
        
        fair_selection = candidate_pool[:num_to_select]
        selected_clients.extend(fair_selection)

    # 4. Update selection counts for fairness tracking
    for cid in selected_clients:
        client_selection_counts[cid] = client_selection_counts.get(cid, 0) + 1

    print(f"[SCHEDULER] Advanced selection complete. Selected clients: {selected_clients}")
    return selected_clients