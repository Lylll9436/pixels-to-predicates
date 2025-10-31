#!/usr/bin/env python3
"""
Relation pattern visualization and statistics script

Features:
1. Read GraphMAE test set prediction results, select TOP-K high/low score samples for each of the six perception dimensions.
2. Build relation network graphs from corresponding scene graphs (JSON), output visualizations for papers.
3. Count triplet frequencies in high/low score samples, output CSV for difference analysis.

Plotting style references Nature journal color scheme.
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, FancyArrowPatch
import networkx as nx
import pandas as pd
import seaborn as sns

PERCEPTION_DIMENSIONS = ["safety", "lively", "boring", "wealthy", "depressing", "beautiful"]
DEFAULT_PREDICTION_PATH = Path("result/predictions_graphmae.csv")
DEFAULT_SPLIT_PATH = Path("output/evaluation/dataset_split.json")
SCENE_GRAPH_DIR = Path("output/stage_03_scene_graphs")
NATURE_PALETTE = ['#1f4e5f', '#326273', '#488a99', '#6ba292', '#a3c9a8', '#f0a202', '#d95d39', '#202c39']


def apply_nature_style() -> None:
    """Configure matplotlib/seaborn to mimic Nature/Sci visual style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#2a2a2a",
        "axes.labelcolor": "#1a1a1a",
        "text.color": "#1a1a1a",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "grid.color": "#d0d0d0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.2,
    })
    sns.set_style("whitegrid")



def aggregate_pairwise_predictions(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {'left_id', 'right_id', 'category'}
    prob_col = None
    for col in df.columns:
        if col.lower().startswith('prob'):
            prob_col = col
            break
    if not required_cols.issubset(df.columns) or prob_col is None:
        raise ValueError('Pairwise prediction file missing required columns (left_id/right_id/category/prob*)')

    categories = sorted(df['category'].unique())
    scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        cat = row['category']
        left = str(row['left_id'])
        right = str(row['right_id'])
        prob = float(row[prob_col])
        prob = max(min(prob, 1.0), 0.0)
        scores[left][cat].append(prob)
        counts[left][cat] += 1
        scores[right][cat].append(1.0 - prob)
        counts[right][cat] += 1

    records = []
    for image_id, cat_scores in scores.items():
        record: Dict[str, float] = {'image_id': image_id}
        for cat in categories:
            values = cat_scores.get(cat)
            if values:
                record[cat] = float(np.mean(values))
                record[f'{cat}_count'] = counts[image_id][cat]
            else:
                record[cat] = float('nan')
                record[f'{cat}_count'] = 0
        records.append(record)

    return pd.DataFrame(records)

def load_predictions(pred_path: Path) -> pd.DataFrame:
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {pred_path}")
    df = pd.read_csv(pred_path)
    if "image_id" in df.columns:
        return df
    aggregated = aggregate_pairwise_predictions(df)
    if aggregated.empty:
        raise ValueError("Prediction file missing image_id column, and cannot aggregate from pairwise data")
    return aggregated


def load_test_ids(split_path: Path) -> List[str]:
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split file does not exist: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    test_ids = data.get("test_ids")
    if not test_ids:
        raise ValueError("test_ids not found in dataset split file")
    return test_ids


def select_extreme_samples(df: pd.DataFrame, dimension: str, top_k: int, test_ids: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[df["image_id"].isin(test_ids)].copy()
    if subset.empty:
        print(f'[WARN] Test set has no matching image_id, using full dataset instead.')
        subset = df.copy()
    if dimension not in subset.columns:
        raise ValueError(f"Prediction file missing dimension column: {dimension}")
    count_col = f"{dimension}_count"
    if count_col in subset.columns:
        subset = subset[subset[count_col] > 0]
    subset = subset.dropna(subset=[dimension])
    if subset.empty:
        raise ValueError(f"No available samples for dimension {dimension} in test set")
    top = subset.nlargest(top_k, dimension)
    bottom = subset.nsmallest(top_k, dimension)
    return top, bottom


def load_scene_graph(file_id: str) -> Optional[Dict]:
    graph_file = SCENE_GRAPH_DIR / f"graph_{file_id}.json"
    if not graph_file.exists():
        return None
    try:
        with graph_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None




def get_entity_label(entity: Dict) -> str:
    for key in ("type", "class", "label", "name", "category"):
        value = entity.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "unknown"


def get_relation_endpoints(rel: Dict) -> Tuple[Optional[str], Optional[str], str]:
    subj = rel.get("subject") or rel.get("subject_id")
    obj = rel.get("object") or rel.get("object_id")
    predicate = rel.get("predicate") or rel.get("relation") or rel.get("type") or "related_to"
    return subj, obj, predicate

def build_triplets(graph_json: Dict) -> List[Tuple[str, str, str]]:
    entities = {ent.get("id"): get_entity_label(ent) for ent in graph_json.get("entities", []) if ent.get("id")}
    triplets: List[Tuple[str, str, str]] = []
    for rel in graph_json.get("relations", []):
        subj_id, obj_id, predicate = get_relation_endpoints(rel)
        subj_label = entities.get(subj_id, "unknown")
        obj_label = entities.get(obj_id, "unknown")
        triplets.append((subj_label, predicate, obj_label))
    return triplets


def draw_relation_graph(graph_json: Dict, title: Optional[str], output_path: Path, show_title: bool = True) -> None:
    entities = graph_json.get('entities', [])
    relations = graph_json.get('relations', [])
    if not entities:
        return

    nodes = []
    node_labels_map: Dict[str, str] = {}
    for ent in entities:
        node_id = ent.get('id')
        if not node_id:
            continue
        label = get_entity_label(ent)
        nodes.append(node_id)
        node_labels_map[node_id] = label

    if not nodes:
        return

    # Ensure stable order, sort by label group then by node ID to reduce crossings
    nodes_sorted = sorted(nodes, key=lambda nid: (node_labels_map.get(nid, ''), nid))

    radius = 3.0
    angles = {node: 2 * np.pi * idx / len(nodes_sorted) for idx, node in enumerate(nodes_sorted)}
    node_positions = {node: (radius * np.cos(angles[node]), radius * np.sin(angles[node])) for node in nodes_sorted}

    unique_labels = sorted(list({node_labels_map[n] for n in nodes_sorted}))
    color_map = {label: NATURE_PALETTE[i % len(NATURE_PALETTE)] for i, label in enumerate(unique_labels)}
    node_colors = [color_map[node_labels_map[n]] for n in nodes_sorted]

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect('equal')
    fig.subplots_adjust(left=0.07, right=0.72, top=0.88, bottom=0.12)
    node_size = 1300

    for node, color in zip(nodes_sorted, node_colors):
        x, y = node_positions[node]
        ax.scatter(x, y, s=node_size, color=color, edgecolor='#1a1a1a', linewidth=1.0, zorder=3)

    limit = radius + 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    label_radius = radius + 0.48
    for node in nodes_sorted:
        theta = angles[node]
        x = label_radius * np.cos(theta)
        y = label_radius * np.sin(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        if cos_theta > 0.2:
            ha = 'left'
        elif cos_theta < -0.2:
            ha = 'right'
        else:
            ha = 'center'
        if sin_theta > 0.2:
            va = 'bottom'
        elif sin_theta < -0.2:
            va = 'top'
        else:
            va = 'center'
        ax.text(x, y, node_labels_map[node], fontsize=11, fontweight='bold', ha=ha, va=va, color='#1a1a1a')

    # Aggregate multiple relations for same subject-object pair
    relation_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for rel in relations:
        subj_id, obj_id, predicate = get_relation_endpoints(rel)
        if subj_id not in node_positions or obj_id not in node_positions or subj_id == obj_id:
            continue
        relation_counts[(subj_id, obj_id, predicate)] += 1

    for (subj_id, obj_id, predicate), count in relation_counts.items():
        theta_u = angles[subj_id]
        theta_v = angles[obj_id]
        delta = (theta_v - theta_u + np.pi) % (2 * np.pi) - np.pi
        if abs(delta) < 1e-4:
            delta = 1e-4
        direction = 1 if delta > 0 else -1
        curvature = 0.4 * direction * max(abs(delta) / np.pi, 0.2) * (1 + 0.15 * min(count - 1, 3))

        start = np.array(node_positions[subj_id])
        end = np.array(node_positions[obj_id])
        if np.allclose(start, end):
            continue

        chord_vec = end - start
        chord_mid = start + 0.5 * chord_vec
        chord_len = math.hypot(chord_vec[0], chord_vec[1]) or 1.0
        normal = np.array([-chord_vec[1], chord_vec[0]]) / chord_len
        control_point = chord_mid + normal * curvature

        path = MplPath([tuple(start), tuple(control_point), tuple(end)],
                       [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        subj_color = color_map[node_labels_map[subj_id]]
        line_width = 2.6 + 0.4 * min(count - 1, 4)
        patch = PathPatch(
            path,
            facecolor='none',
            edgecolor=subj_color,
            lw=line_width,
            alpha=0.8,
            zorder=2,
            capstyle='round',
            joinstyle='round'
        )
        ax.add_patch(patch)

        # Calculate label position: approximate midpoint along Bezier curve
        mid_t = 0.5
        bx = (1 - mid_t)**2 * start[0] + 2 * (1 - mid_t) * mid_t * control_point[0] + mid_t**2 * end[0]
        by = (1 - mid_t)**2 * start[1] + 2 * (1 - mid_t) * mid_t * control_point[1] + mid_t**2 * end[1]
        label_offset = 0.35 * curvature
        normal_x = -(end[1] - start[1])
        normal_y = end[0] - start[0]
        norm_len = math.hypot(normal_x, normal_y) or 1.0
        label_x = bx + (normal_x / norm_len) * label_offset
        label_y = by + (normal_y / norm_len) * label_offset
        ax.text(
            label_x,
            label_y,
            predicate,
            fontsize=10,
            color=subj_color,
            ha='center',
            va='center',
            zorder=4,
            bbox={'boxstyle': 'round,pad=0.2', 'fc': 'white', 'ec': 'none', 'alpha': 0.7}
        )

    if unique_labels:
        handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=color_map[label], label=label) for label in sorted(unique_labels)]
        ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.25, -0.02), frameon=True, title="Entities")

    if show_title and title:
        ax.set_title(title)
    ax.axis('off')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=500, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)

def aggregate_triplets(file_ids: Iterable[str]) -> Counter:
    counter: Counter = Counter()
    for fid in file_ids:
        graph_json = load_scene_graph(fid)
        if not graph_json:
            continue
        for triplet in build_triplets(graph_json):
            counter[triplet] += 1
    return counter


def save_triplet_stats(counter_high: Counter, counter_low: Counter, output_path: Path, top_n: int = 30) -> None:
    triplets = set(counter_high) | set(counter_low)
    rows = []
    for triplet in triplets:
        rows.append({
            "subject": triplet[0],
            "predicate": triplet[1],
            "object": triplet[2],
            "high_count": counter_high.get(triplet, 0),
            "low_count": counter_low.get(triplet, 0),
            "delta": counter_high.get(triplet, 0) - counter_low.get(triplet, 0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("delta", ascending=False, inplace=True)
    df.head(top_n).to_csv(output_path, index=False, encoding="utf-8-sig")


def create_triplet_barplot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="delta", y="relation", palette="crest", ax=ax)
    ax.set_xlabel("High - Low Frequency")
    ax.set_ylabel("Triplet")
    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def create_combined_chord_grid(summary_df: pd.DataFrame, output_dir: Path, top_k: int) -> Optional[Path]:
    if summary_df.empty:
        return None

    dimensions = [dim for dim in PERCEPTION_DIMENSIONS if dim in summary_df["dimension"].unique()]
    if not dimensions:
        return None

    nrows = len(dimensions)
    ncols = max(2, top_k * 2)
    fig_width = 2.4 * ncols
    fig_height = 2.4 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, dim in enumerate(dimensions):
        low_df = summary_df[(summary_df["dimension"] == dim) & (summary_df["group"] == "low")]
        high_df = summary_df[(summary_df["dimension"] == dim) & (summary_df["group"] == "high")]

        low_paths: List[Path] = []
        for candidate in low_df.sort_values("score", ascending=True)["output_notitle"].tolist():
            path = Path(candidate)
            if path.exists():
                low_paths.append(path)
            if len(low_paths) >= top_k:
                break

        high_paths: List[Path] = []
        for candidate in high_df.sort_values("score", ascending=True)["output_notitle"].tolist():
            path = Path(candidate)
            if path.exists():
                high_paths.append(path)
            if len(high_paths) >= top_k:
                break

        for col_idx in range(ncols):
            axes[row_idx, col_idx].axis("off")

        for col_idx, img_path in enumerate(low_paths):
            ax = axes[row_idx, col_idx]
            ax.imshow(plt.imread(img_path))
            ax.axis("off")

        high_offset = top_k
        for col_idx, img_path in enumerate(high_paths):
            target_col = min(high_offset + col_idx, ncols - 1)
            ax = axes[row_idx, target_col]
            ax.imshow(plt.imread(img_path))
            ax.axis("off")

    plt.subplots_adjust(left=0.14, right=0.98, bottom=0.05, top=0.9, wspace=0.05, hspace=0.18)

    for idx, dim in enumerate(dimensions):
        y = 0.9 - (idx + 0.5) / nrows * 0.8
        fig.text(0.08, y, dim, fontsize=14, fontweight="bold", ha="right", va="center")

    fig.text(0.18, 0.94, "low", fontsize=14, fontweight="bold", ha="left", va="center")
    fig.text(0.82, 0.94, "high", fontsize=14, fontweight="bold", ha="right", va="center")
    arrow = FancyArrowPatch(
        (0.22, 0.935),
        (0.78, 0.935),
        transform=fig.transFigure,
        arrowstyle="->",
        linewidth=1.5,
        color="#333333",
        mutation_scale=16
    )
    fig.add_artist(arrow)

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_dir / f"relation_chord_grid_top{top_k}.png"
    fig.savefig(combined_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return combined_path


def main():
    parser = argparse.ArgumentParser(description="GraphMAE test set relation pattern visualization")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTION_PATH, help="GraphMAE prediction results CSV")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH, help="Dataset split JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("output/evaluation/rel_visual"), help="Output directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of high/low score samples")
    parser.add_argument("--triplet-top", type=int, default=30, help="Top N triplets to export in statistics")
    args = parser.parse_args()

    apply_nature_style()

    predictions = load_predictions(args.predictions)
    test_ids = load_test_ids(args.split)
    out_dir = args.output_dir

    summary_rows = []

    for dim in PERCEPTION_DIMENSIONS:
        try:
            top_df, bottom_df = select_extreme_samples(predictions, dim, args.top_k, test_ids)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue

        group_info = [("high", top_df), ("low", bottom_df)]

        for group_name, df_group in group_info:
            for idx, row in df_group.iterrows():
                fid = row["image_id"]
                graph_json = load_scene_graph(fid)
                if not graph_json:
                    print(f"[WARN] Missing scene graph: {fid}")
                    continue
                title = f"{dim.capitalize()} - {group_name.capitalize()} #{len(summary_rows) + 1}"
                filename = f"{dim}_{group_name}_{fid}.png"
                graph_path = out_dir / "graphs" / filename
                draw_relation_graph(graph_json, title, graph_path, show_title=True)
                graph_notitle_path = out_dir / "graphs_notitle" / filename
                draw_relation_graph(graph_json, title, graph_notitle_path, show_title=False)
                summary_rows.append({
                    "dimension": dim,
                    "group": group_name,
                    "image_id": fid,
                    "score": row[dim],
                    "output": str(graph_path),
                    "output_notitle": str(graph_notitle_path),
                })

        # triplet statistics
        high_ids = top_df["image_id"].tolist()
        low_ids = bottom_df["image_id"].tolist()
        counter_high = aggregate_triplets(high_ids)
        counter_low = aggregate_triplets(low_ids)

        triplet_csv = out_dir / "triplets" / f"{dim}_triplets.csv"
        save_triplet_stats(counter_high, counter_low, triplet_csv, top_n=args.triplet_top)

        # barplot for top delta
        if triplet_csv.exists():
            df_triplet = pd.read_csv(triplet_csv)
            if not df_triplet.empty:
                df_triplet["relation"] = df_triplet.apply(lambda r: f"{r['subject']}|{r['predicate']}|{r['object']}", axis=1)
                barplot_path = out_dir / "triplets" / f"{dim}_triplets_delta.png"
                create_triplet_barplot(df_triplet.head(15), barplot_path, f"{dim.capitalize()} Triplet Frequency Δ")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "graph_summary.csv", index=False, encoding="utf-8-sig")
        combined_path = create_combined_chord_grid(summary_df, out_dir, top_k=args.top_k)
        print(f"[INFO] Generated {len(summary_rows)} relation network graphs in total, statistics saved to {out_dir}")
        if combined_path:
            print(f"[INFO] Combined visualization exported to {combined_path}")
    else:
        print("[WARN] No images generated, check if data is complete.")


if __name__ == "__main__":
    main()

