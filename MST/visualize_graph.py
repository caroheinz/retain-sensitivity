
import networkx as nx
import matplotlib.pyplot as plt
import os
from calc_retain_sens import get_realistic_subgraph, calculate_mst_retain_sensitivity



def visualize_experiment(G, MST, rs_edge, rs_value):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))

    # A. Draw ALL edges (Background - Light Gray)
    # We exclude MST edges here to avoid drawing over them twice
    mst_edges_set = set(tuple(sorted((u, v))) for u, v in MST.edges())
    non_mst_edges = [e for e in G.edges() if tuple(sorted(e)) not in mst_edges_set]

    nx.draw_networkx_edges(G, pos, edgelist=non_mst_edges,
                           edge_color='lightgray', width=1, alpha=0.5)

    # B. Draw MST edges (Safe - Blue)
    # Filter out the RS edge so we don't draw it blue
    safe_mst_edges = [e for e in MST.edges()
                      if tuple(sorted(e)) != tuple(sorted(rs_edge))]

    nx.draw_networkx_edges(G, pos, edgelist=safe_mst_edges,
                           edge_color='blue', width=2.5, label='MST Edges')

    # C. Draw RS Edge (Vulnerable/Replaced - Red)
    if rs_edge:
        nx.draw_networkx_edges(G, pos, edgelist=[rs_edge],
                               edge_color='red', width=4, label=f'RS Edge (w={rs_value})')

    # D. Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color='black')

    # E. Draw Labels (For ALL edges now)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    # 1. Draw standard labels (Black) for everything EXCEPT the RS edge
    normal_labels = {k: v for k, v in edge_labels.items() if tuple(sorted(k)) != tuple(sorted(rs_edge if rs_edge else (None, None)))}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=normal_labels, font_size=8, font_color='black')

    # 2. Draw RS label (Red & Bold) specifically
    if rs_edge:
        # We make this one slightly larger and red
        rs_label = {rs_edge: rs_value}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=rs_label, font_size=12, font_color='red', font_weight='bold')

    plt.title(f"Retain Sensitivity Visualization\nRS Value: {rs_value}", fontsize=14)
    plt.legend(loc='upper right')
    plt.axis('off')
    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)
    plt.locator_params(axis='y', nbins=4)
    plt.savefig(os.path.join(out_dir, "rs_MST_example.png"), dpi=220, bbox_inches="tight")
    plt.show()

def run_single_experiment_visual(full_graph, nodes_n=50):

    print(f"Generating realistic subgraph with {nodes_n} nodes...")

    subG = get_realistic_subgraph(full_graph, nodes_n)

    mst = nx.minimum_spanning_tree(subG)

    max_edge = calculate_mst_retain_sensitivity(subG, mst)
    rs_val = max_edge[2] if max_edge is not None else None
    rs_edge = (max_edge[0], max_edge[1]) if max_edge is not None else None

    print(f"Analysis Complete.")
    print(f"MST Weight: {sum(d['weight'] for u,v,d in mst.edges(data=True))}")
    print(f"Retain Sensitivity: {rs_val}")
    print(f"Vulnerable Edge: {rs_edge}")

    visualize_experiment(subG, mst, rs_edge, rs_val)