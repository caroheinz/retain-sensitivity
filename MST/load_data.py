

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import pandas as pd
import requests
import gzip
import shutil
import os
import random
from pathlib import Path
import io
import urllib.request
import pandas as pd
import networkx as nx
import graph_tool.all as gt


# Load data for graphs and for ERM experiments
DATA_URL = "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"
DATA_DIR = Path(".")
DATA_PATH = DATA_DIR / "data/soc-sign-bitcoinotc.csv.gz"

#load bitcoin data

def download_bitcoin_otc(path: Path = DATA_PATH, data_url=DATA_URL) -> Path:
    if path.exists():
        return path
    with urllib.request.urlopen(data_url) as resp:
        raw = resp.read()
    path.write_bytes(raw)
    return path


def load_bitcoin_otc(path: Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        path = download_bitcoin_otc(path)
    with gzip.open(path, "rb") as f:
        buf = io.BytesIO(f.read())
    df = pd.read_csv(
        buf,
        header=None,
        names=["source", "target", "weight", "time"],
    )
    # drop time
    df = df.drop(columns=["time"])
    # Add 10 to all weights to make them non negative
    df['weight'] = df['weight'] + 10
    
    min_weight = df['weight'].min()
    max_weight = df['weight'].max()

    # Create undirected graph
    DiG = nx.from_pandas_edgelist(
        df,
        source="source",
        target="target",
        edge_attr=["weight"],
        create_using=nx.DiGraph,
    )
    undirected_G = DiG.to_undirected()


    n_total = undirected_G.number_of_nodes()
    weight_stats = (min_weight, max_weight)


    return undirected_G, n_total, weight_stats


# Load graph-tool datasets

def load_graph_tool_dataset(name, weight_key=None, undirected=True):
    """
    Load a graph-tool dataset and return (G, n_total, weight_stats).
    weight_key: edge property to use as weight (defaults to 'count' if present, else 'weight' if present).
    """

    g_gt = gt.collection.ns[name]
    edges = g_gt.get_edges()
    source_ids = edges[:, 0]
    target_ids = edges[:, 1]

    if weight_key is None:
        if 'count' in g_gt.ep:
            weight_key = 'count'
        elif 'weight' in g_gt.ep:
            weight_key = 'weight'

    if weight_key is not None and weight_key in g_gt.ep:
        weights = g_gt.ep[weight_key].a
    else:
        weights = None

    df = pd.DataFrame({
        'source': source_ids,
        'target': target_ids,
    })
    if weights is not None:
        df['weight'] = weights
    else:
        df['weight'] = 1.0

    df = df[df['source'] != df['target']].copy()
    df = df.groupby(['source', 'target'])['weight'].sum().reset_index()

    min_weight = df['weight'].min()
    max_weight = df['weight'].max()

    G = nx.from_pandas_edgelist(
        df,
        source='source',
        target='target',
        edge_attr=['weight'],
        create_using=nx.Graph() if undirected else nx.DiGraph()
    )

    n_total = G.number_of_nodes()
    weight_stats = (min_weight, max_weight)
    return G, n_total, weight_stats




## Load requested graphs

def load_requested_graphs(dataset_dict):
    """
    Load requested graphs based on dataset_dict
    returns updated dataset_dict with loaded graphs
    """
    for dataset_name in dataset_dict.keys():
        if dataset_name == 'bitcoin_otc':
            full_bitcoin_graph,_,_= load_bitcoin_otc()
            dataset_dict[dataset_name]["loaded"] = full_bitcoin_graph
        else:
            G,_,_ = load_graph_tool_dataset(name=dataset_name)
            dataset_dict[dataset_name]["loaded"] = G
    return dataset_dict

