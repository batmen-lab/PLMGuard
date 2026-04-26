import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.plt_utils import get_method_name

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def _extract_exp1_metric(url):
    metric_df = pd.read_csv(url)
    metric_df = metric_df.T
    metric_df.columns = [
        'exp1_auprc_origin_to_t1',
        'exp1_auprc_t1_to_t2',
        'exp1_auprc_t2_to_t3',
        'exp1_auprc_t3_to_t4',
        'exp1_auprc_t4_to_t5',
    ]
    return metric_df

def _extract_exp2_metric(url):
    metric_df = pd.read_csv(url)
    metric_df = metric_df.set_index('data_type')
    metric_df = metric_df.filter(regex='mut_mean')
    metric_df.columns = ['exp2_mean_auroc']
    metric_df.index.name = None
    return metric_df

def _extract_exp3_metric(url):
    metric_df = pd.read_csv(url)
    metric_df = metric_df.pivot(
        index="data_type",
        columns="decoy_type",
        values="avg_norm_dist"
    )
    metric_df.columns.name = None
    metric_df.index.name = None
    metric_df.columns = [
        'exp3_avg_norm_dist_doubshuf',
        'exp3_avg_norm_dist_doubself',
    ]
    return metric_df

def _extract_exp4_metric(url):
    metric_df = pd.read_csv(url)
    metric_df = metric_df.set_index('data_type')
    metric_df.columns = ['exp4_monotonicity_loss']
    metric_df.index.name = None
    return metric_df

def _extract_exp5_metric(url):
    metric_df = pd.read_csv(url)
    metric_df["abs_auroc_diff"] = (metric_df["auroc"] - 0.5).abs()
    metric_df = metric_df.groupby("data_type")["abs_auroc_diff"].mean().to_frame()
    metric_df = metric_df[metric_df.index != "mmseq2"]
    metric_df.columns = ["exp5_avg_abs_auroc_diff"]
    metric_df.index.name = None
    return metric_df

def _extract_exp6_metric(url):
    metric_df = pd.read_csv(url)
    metric_df["abs_auroc_diff"] = (metric_df["auroc"] - 0.5).abs()
    metric_df = metric_df.groupby("data_type")["abs_auroc_diff"].mean().to_frame()
    metric_df.columns = ["exp6_avg_abs_auroc_diff"]
    for method in ['blastp', 'mmseq2', 'diamond']:
        if method not in metric_df.index:
            metric_df.loc[method] = 0.0
    metric_df.index.name = None
    return metric_df


def extract_metrics(metric_urls):
    dfs = [extractor(url) for extractor, url in zip(_EXTRACTORS, metric_urls)]
    metric_df = pd.concat(dfs, axis=1)
    metric_df["method_id"] = metric_df.index
    metric_df["method_name"] = metric_df["method_id"].map(get_method_name)

    exp_groups = sorted({c.split("_")[0] for c in metric_df.columns if c.startswith("exp")})

    def melt_group(group):
        cols = [c for c in metric_df.columns if c.startswith(f"{group}_")]
        sub = metric_df[["method_id", "method_name"] + cols].melt(
            id_vars=["method_id", "method_name"], var_name="metric", value_name="value"
        )
        sub["exp_group"] = group
        return sub

    long_df = pd.concat(
        [melt_group(g) for g in exp_groups],
        axis=0,
        ignore_index=True,
    )

    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])

    # Normalize for color per metric (higher = 1.0)
    long_df["color_norm"] = 0.5
    for _, group in long_df.groupby("metric"):
        vals = group["value"].astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            long_df.loc[group.index, "color_norm"] = (vals - vmin) / (vmax - vmin)

    # Normalize for size: some exps grouped together, others per metric
    long_df["size_norm"] = 0.5
    for exp_group in _SIZE_NORM_BY_EXP:
        idx = long_df["exp_group"] == exp_group
        if idx.any():
            vals = long_df.loc[idx, "value"].astype(float)
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                long_df.loc[idx, "size_norm"] = (vals - vmin) / (vmax - vmin)
    per_metric_idx = long_df["exp_group"].isin(set(exp_groups) - _SIZE_NORM_BY_EXP)
    for _, group in long_df[per_metric_idx].groupby("metric"):
        vals = group["value"].astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            long_df.loc[group.index, "size_norm"] = (vals - vmin) / (vmax - vmin)

    return long_df


def plot_bubble_overall(long_df, out_path):
    exp_groups = sorted(long_df["exp_group"].unique())
    exp_cols = {g: sorted(long_df[long_df["exp_group"] == g]["metric"].unique()) for g in exp_groups}

    def color_for(row):
        n = row["color_norm"]
        if row["exp_group"] in _WARM_EXP_GROUPS:
            return plt.cm.Reds(n)
        return plt.cm.Blues(1.0 - n)

    long_df = long_df.copy()
    long_df["color"] = long_df.apply(color_for, axis=1)
    long_df["size"] = 30 + long_df["size_norm"] * (260 - 30)

    # Fixed method order
    fixed_order = ['blastp', 'diamond', 'mmseq2', 'near', 'dctdomain', 'dhr', 'plm', 'tmvec']
    available_ids = set(long_df["method_id"].unique())
    method_order = [get_method_name(m) for m in fixed_order if m in available_ids]

    metric_order = [m for g in exp_groups for m in exp_cols[g]]

    # Compute start/end index of each exp group within metric_order
    boundaries = {}
    pos = 0
    for g in exp_groups:
        boundaries[g] = (pos, pos + len(exp_cols[g]) - 1)
        pos += len(exp_cols[g])

    fig, ax = plt.subplots(figsize=(1.0 * len(metric_order) + 4, 0.45 * len(method_order) + 2.4))
    gap = 0.55
    gap_map = {}
    for g in exp_groups:
        gap_map[boundaries[g][1]] = gap

    offset = 0.0
    x_positions = []
    for i, _ in enumerate(metric_order):
        x_positions.append(i + offset)
        if i in gap_map:
            offset += gap_map[i]
    x_map = {m: x_positions[i] for i, m in enumerate(metric_order)}
    y_map = {m: i for i, m in enumerate(method_order)}

    for idx in range(len(method_order)):
        if idx % 2 == 0:
            ax.axhspan(idx - 0.5, idx + 0.5, xmin=0, xmax=1, color="#EEF1FA", alpha=1.0, zorder=0)

    display_df = long_df[~long_df.apply(
        lambda r: r["method_id"] in _DISPLAY_EXCLUDE.get(r["exp_group"], set()), axis=1
    )]
    displayed_cells = set(zip(display_df["method_name"], display_df["metric"]))

    for _, row in display_df.iterrows():
        if row["metric"] not in x_map:
            continue
        y = y_map.get(row["method_name"])
        if y is None:
            continue
        ax.scatter(x_map[row["metric"]], y, s=row["size"], color=row["color"], edgecolors="black", linewidths=0.2)

    for method_name in method_order:
        for metric in metric_order:
            if (method_name, metric) not in displayed_cells:
                ax.text(
                    x_map[metric], y_map[method_name], "N/A",
                    ha="center", va="center", fontsize=6, color="gray", style="italic",
                )

    tick_labels = [_TICK_LABELS.get(m, m) for m in metric_order]
    ax.set_xticks(list(x_map.values()))
    ax.set_xticklabels(tick_labels, rotation=45, ha="left", fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", labeltop=True, labelbottom=False)
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(method_order, ha="left")
    ax.tick_params(axis="y", pad=70)
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Overall Performance Bubble Plot", pad=46)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)

    for g in exp_groups:
        start, end = boundaries[g]
        x_start = x_map[metric_order[start]] - 0.45
        x_end = x_map[metric_order[end]] + 0.45
        x_center = (x_start + x_end) / 2
        ax.text(
            x_center, 1.32, _GROUP_LABEL_TEXTS.get(g, g),
            ha="center", va="bottom", fontsize=8,
            transform=ax.get_xaxis_transform(),
        )
        ax.plot(
            [x_start, x_end], [1.26, 1.26],
            transform=ax.get_xaxis_transform(),
            color="black", linewidth=0.8, clip_on=False,
        )

    ax.set_xlim(-1.0, max(x_positions) + 1.0)

    sm_warm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(0, 1))
    sm_warm.set_array([])
    sm_cool = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(0, 1))
    sm_cool.set_array([])
    cax1 = fig.add_axes([0.82, 0.05, 0.04, 0.01])
    cax2 = fig.add_axes([0.82, 0.02, 0.04, 0.01])
    cbar1 = fig.colorbar(sm_warm, cax=cax1, orientation="horizontal")
    cbar2 = fig.colorbar(sm_cool, cax=cax2, orientation="horizontal")
    cbar1.set_ticks([])
    cbar2.set_ticks([])

    fig.tight_layout(rect=[0.08, 0.0, 0.82, 0.84])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# --- per-experiment configuration (extend here when adding a new experiment) ---

_EXTRACTORS = [
    _extract_exp1_metric,
    _extract_exp2_metric,
    _extract_exp3_metric,
    _extract_exp4_metric,
    _extract_exp5_metric,
    _extract_exp6_metric,
]

# methods kept in data only for size scaling and hidden from the plot per exp group
_DISPLAY_EXCLUDE = {
    "exp6": {"blastp", "diamond", "mmseq2"},
}

# exps whose size is normalized across all metrics in the group (vs. per metric)
_SIZE_NORM_BY_EXP = {"exp1", "exp5", "exp6"}

# exps where higher value is better → warm (Reds) colormap; others → cool (Blues)
_WARM_EXP_GROUPS = {"exp1", "exp2"}

_GROUP_LABEL_TEXTS = {
    "exp1": "Evolutionary\nplausibility",
    "exp2": "Structure\nconsistency",
    "exp3": "Redundancy\nstability",
    "exp4": "Similarity\nmonotonicity",
    "exp5": "Decoy\nsensitivity",
    "exp6": "Representation\nreliability",
}

_TICK_LABELS = {
    "exp1_auprc_origin_to_t1":       "AUPRC\n(Origin to t1)",
    "exp1_auprc_t1_to_t2":           "AUPRC\n(t1 to t2)",
    "exp1_auprc_t2_to_t3":           "AUPRC\n(t2 to t3)",
    "exp1_auprc_t3_to_t4":           "AUPRC\n(t3 to t4)",
    "exp1_auprc_t4_to_t5":           "AUPRC\n(t4 to t5)",
    "exp2_mean_auroc":               "Mean\nAUROC",
    "exp3_avg_norm_dist_doubshuf":   "Avg Norm\nDistance\n(Doubshuf)",
    "exp3_avg_norm_dist_doubself":   "Avg Norm\nDistance\n(Doubself)",
    "exp4_monotonicity_loss":        "Monotonicity\nLoss",
    "exp5_avg_abs_auroc_diff":       "Avg Abs\nAUROC Diff",
    "exp6_avg_abs_auroc_diff":       "Avg Abs\nAUROC Diff",
}


def main():
    metric_urls = [
        "../results/check_bio_evolution/data/metrics.csv",
        "../results/check_bio_structure/data/metrics.csv",
        "../results/check_pert_doublen/data/metrics.csv",
        "../results/check_pert_truncation/data/metrics.csv",
        "../results/check_perm_data/data/metrics.csv",
        "../results/check_perm_model/data/metrics.csv",
    ]
    save_dir = "../results/overall_performance"
    os.makedirs(save_dir, exist_ok=True)
    long_df = extract_metrics(metric_urls)
    plot_bubble_overall(long_df, out_path=os.path.join(save_dir, "overall_performance_bubble.pdf"))


if __name__ == "__main__":
    main()

# Command to run the code:
# python sanity_check/overall_performance.py
