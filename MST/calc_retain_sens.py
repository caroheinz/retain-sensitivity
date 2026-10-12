import networkx as nx
import pandas as pd
from tqdm import tqdm
import random


def get_realistic_subgraph(full_graph, n_nodes):
    # 1. Pick a random start node
    start_node = random.choice(list(full_graph.nodes()))

    # 2. Perform BFS
    subgraph_nodes = set()
    queue = [start_node]
    subgraph_nodes.add(start_node)

    while len(subgraph_nodes) < n_nodes and queue:
        current = queue.pop(0)
        for neighbor in full_graph.neighbors(current):
            if neighbor not in subgraph_nodes:
                subgraph_nodes.add(neighbor)
                queue.append(neighbor)
                if len(subgraph_nodes) >= n_nodes:
                    break
    if len(subgraph_nodes) < n_nodes:
        return None
    # 3. Create the subgraph
    subgraph = full_graph.subgraph(subgraph_nodes).copy()

    return subgraph

def compute_weight_range(G):
    weights = [d.get('weight', 0) for _, _, d in G.edges(data=True)]
    if not weights:
        raise ValueError('Graph has no edge weights')
    return max(weights) - min(weights)

def calculate_mst_retain_sensitivity(G, T):
    """
    Calculates the Retain Sensitivity for the MST of a given graph G.
    Logic:
    1. Compute MST of G.
    2. For each edge (u, v) in MST:
       - Remove (u, v) to split vertices into two sets (S, V-S).
       - Check if the cut (S, V-S) is 'saturated' in the original graph G.
       - Saturated means all possible edges between S and V-S already exist.
       - If NOT saturated, an adversary could add a 0-weight edge there,
         reducing the MST cost by exactly weight(u, v).
    3. RS is the maximum weight among all such unsaturated cuts.

    Returns:
        (u, v, w) for the MST edge achieving the maximum RS among unsaturated cuts,
        or None if all such cuts are saturated.
    """

    max_sensitivity = 0.0
    max_edge = None

    mst_edges = list(T.edges(data=True))

    for u, v, data in mst_edges:
        current_weight = data['weight']

        # 2. Virtual Removal: Temporarily remove the edge to identify the cut
        T.remove_edge(u, v)

        # Identify the two connected components (S and V\S) created by the removal
        # Since T was a tree, removing one edge splits it into exactly 2 components
        components = list(nx.connected_components(T))

        if len(components) == 2:
            S, V_minus_S = components

            # 3. Check Saturation
            # Calculate maximum possible edges crossing this cut
            max_possible_edges = len(S) * len(V_minus_S)

            # Calculate actual existing edges crossing this cut in the original graph G
            # nx.edge_boundary returns all edges in G that connect a node in S to a node not in S
            actual_crossing_edges = list(nx.edge_boundary(G, S))
            num_actual_edges = len(actual_crossing_edges)

            # If the cut is NOT saturated, the adversary can introduce a new edge here.
            # This would allow the MST to drop the current edge (u,v).
            if num_actual_edges < max_possible_edges:
                if current_weight > max_sensitivity:
                    max_sensitivity = current_weight
                    max_edge = (u, v, current_weight)

        # Restore the edge for the next iteration
        T.add_edge(u, v, weight=current_weight)

    return max_edge


def run_experiment_n(G, trials=100, nodes=100, dens=0.0):
    data_rows = []
    counter = 0
    all_attempts = 0
    with tqdm(total=trials, desc="Valid Samples Collected") as pbar:
        while counter < trials:
            all_attempts += 1

            subG = get_realistic_subgraph(G, nodes)
            if subG is None:
                continue

            if nx.density(subG) <= dens:
                continue
            gs_value = compute_weight_range(subG)
            mst = nx.minimum_spanning_tree(subG)
            total_mst_weight = sum(d['weight'] for u, v, d in mst.edges(data=True))
            max_edge = calculate_mst_retain_sensitivity(subG, mst)
            rs_value = max_edge[2] if max_edge is not None else None

            row = {
                'sample_id': all_attempts,
                'target_nodes': nodes,
                'num_nodes': subG.number_of_nodes(),
                'num_edges': subG.number_of_edges(),
                'density': nx.density(subG),
                'mst_weight': total_mst_weight,
                'retain_sensitivity': rs_value,
                'GS': gs_value,
                'RS_over_GS': rs_value / gs_value if rs_value is not None and gs_value != 0 else float('inf')
            }
            data_rows.append(row)

            counter += 1
            pbar.update(1)

    df = pd.DataFrame(data_rows)
    return df



# take dictionary with df_name and loaded graph, run experiment and save results to csv

def run_experiment_and_save(dataset_dict, trials=100, nodes=100, dens=0.0):
    for key, value in dataset_dict.items():
        if value['loaded'] is not None:
            print(f"Running experiment for {key}...")
            df_results = run_experiment_n(value['loaded'], trials=trials, nodes=nodes, dens=dens)
            df_name = value["file"]
            df_results.to_csv(f'results/{df_name}', index=False)
            print(f"Results saved to results/{df_name}")
        else:
            print(f"No loaded graph for {key}, skipping experiment.")

    return dataset_dict
