
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGSIZE = (5,4)

def plot_rs_gs_boxplot_sns(csv_paths, label_map=None):
    """
    Boxplot of RS/GS by dataset using seaborn.
    csv_paths: list of paths to rs_mst_{network}.csv files.
    label_map: optional dict to rename dataset labels.
    """

    records = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if 'RS_over_GS' not in df.columns:
            raise ValueError(f"{path} missing RS_over_GS column")
        name = os.path.basename(path)
        name = name.replace('rs_mst_', '').replace('.csv', '')
        if label_map and name in label_map:
            name = label_map[name]
        for v in df['RS_over_GS'].dropna().values:
            records.append({'dataset': name, 'ratio': v})

    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        raise ValueError('No data to plot')

    order = sorted(plot_df['dataset'].unique())

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, sharey=False)

    sns.boxplot(
        data=plot_df,
        x='dataset',
        y='ratio',
        order=order,
        ax=ax,
        linewidth=4,
        linecolor= "#000000",
        showfliers=True,
        width=.3,
        #flierprops={"marker": "x"},
        boxprops={"facecolor":  "#BDD9FFA0","linewidth": 3},
        medianprops={"color": "#000000", "linewidth": 3},
        whiskerprops={"color": "#000000","linewidth": 3},
    )
    ax.set_ylabel('RS/GS', fontsize=14)
    ax.set_xlabel('Dataset', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)


    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)
    plt.locator_params(axis='y', nbins=4)
    plt.savefig(os.path.join(out_dir, "rs_MST_boxplots.png"), dpi=220, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


