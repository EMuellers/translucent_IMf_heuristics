"""
Compute zero counts per CSV column across a directory tree and plot a heatmap.

Usage:
    python plot_zero_counts_heatmap.py --input processed_results --output plots/zero_counts_heatmap.png --normalize

The script:
- walks `--input` for CSV files
- for each CSV computes number of exact numeric zeros per column (non-numeric values ignored)
- builds a matrix (files x columns) of zero counts
- optionally normalizes counts by number of non-na rows to get proportions
- plots a heatmap and writes to `--output`

Requires: pandas, seaborn, matplotlib
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def find_csv_files(root: Path):
    for p in root.rglob("*.csv"):
        yield p


def count_zeros_in_csv(path: Path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
        return None

    # Convert columns to numeric where possible; count numeric zeros only
    zero_counts = {}
    non_na_counts = {}
    for col in df.columns:
        s = df[col]
        # attempt numeric conversion
        s_num = pd.to_numeric(s, errors='coerce')
        # count zeros among numeric values
        zeros = (s_num == 0).sum()
        non_na = s_num.notna().sum()
        zero_counts[col] = int(zeros)
        non_na_counts[col] = int(non_na)

    return zero_counts, non_na_counts


def build_matrix(results):
    # results: list of (file_label, zero_counts, non_na_counts)
    all_cols = set()
    for _, zc, _ in results:
        all_cols.update(zc.keys())
    all_cols = sorted(all_cols)

    index = [label for label, _, _ in results]
    mat = pd.DataFrame(index=index, columns=all_cols, dtype=float)
    non_na_mat = pd.DataFrame(index=index, columns=all_cols, dtype=float)

    for label, zc, nac in results:
        for col in all_cols:
            mat.at[label, col] = zc.get(col, np.nan)
            non_na_mat.at[label, col] = nac.get(col, np.nan)

    return mat, non_na_mat


def shorten_label(path: Path, root: Path):
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def plot_heatmap(mat: pd.DataFrame, out_path: Path, title=None, vmax=None, cmap='viridis'):
    plt.figure(figsize=(max(8, mat.shape[1] * 0.35), max(6, mat.shape[0] * 0.25)))
    sns.set(style='white')
    ax = sns.heatmap(mat, cmap=cmap, linewidths=0.2, linecolor='gray', cbar_kws={'label': 'zero count'})
    plt.title(title or 'Zero counts heatmap')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser(description='Plot heatmap of zero counts across CSVs')
    p.add_argument('--input', '-i', required=False, default='processed_results', help='Input folder to scan for CSVs')
    p.add_argument('--output', '-o', required=False, default='plots/zero_counts_heatmap.png', help='Output image path')
    p.add_argument('--normalize', action='store_true', help='Normalize by non-NA counts to show proportion of zeros')
    p.add_argument('--top-columns', type=int, default=80, help='Show only the top-N columns by maximum zero count (or proportion)')
    p.add_argument('--min-files', type=int, default=1, help='Minimum number of files a column must appear in to keep it')
    args = p.parse_args()

    root = Path(args.input)
    out = Path(args.output)

    if not root.exists():
        print(f"Input folder {root} does not exist", file=sys.stderr)
        sys.exit(1)

    results = []
    for csv_path in find_csv_files(root):
        res = count_zeros_in_csv(csv_path)
        if res is None:
            continue
        zc, nac = res
        label = shorten_label(csv_path, root)
        results.append((label, zc, nac))

    if not results:
        print(f"No CSV files found under {root}", file=sys.stderr)
        sys.exit(1)

    mat, non_na_mat = build_matrix(results)

    # Optionally normalize by non-NA counts
    if args.normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            mat_norm = mat.div(non_na_mat)
        mat_to_plot = mat_norm
        title = 'Proportion of zeros per column'
        cbar_label = 'proportion zeros'
    else:
        mat_to_plot = mat
        title = 'Zero counts per column'
        cbar_label = 'zero counts'

    # Filter columns by number of files they appear in
    col_presence = mat.notna().sum(axis=0)
    cols_keep = col_presence[col_presence >= args.min_files].index.tolist()
    mat_to_plot = mat_to_plot[cols_keep]

    # Reduce to top columns by max value
    if mat_to_plot.shape[1] > args.top_columns:
        sort_col = mat_to_plot.max(axis=0)
        top_cols = sort_col.sort_values(ascending=False).head(args.top_columns).index.tolist()
        mat_to_plot = mat_to_plot[top_cols]

    # Replace infinite or very large with NaN for plotting safety
    mat_to_plot = mat_to_plot.replace([np.inf, -np.inf], np.nan)

    plot_heatmap(mat_to_plot, out, title=title)
    print(f"Saved heatmap to {out}")


if __name__ == '__main__':
    main()
