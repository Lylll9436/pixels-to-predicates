#!/usr/bin/env python3
"""
Post-training evaluation and visualization for Bradley-Terry comparison model.

Outputs to ./result:
- bradley_terry_comparison_best.pth (copied if exists)
- metrics.json (overall + per-category)
- predictions.csv (detailed pairs)
- category_auc.png, roc_curve.png, pr_curve.png, confusion_matrix.png
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from shutil import copy2

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.linear_model import LogisticRegression


NATURE_PALETTE = ['#1f4e5f', '#326273', '#488a99', '#6ba292', '#a3c9a8']
ACCENT_COLOR = '#d1495b'
BACKGROUND_COLOR = '#f4f4f2'


def apply_nature_style() -> None:
    """Configure matplotlib/seaborn to mimic Nature/Sci visual style."""
    plt.rcParams.update({
        'figure.facecolor': BACKGROUND_COLOR,
        'axes.facecolor': BACKGROUND_COLOR,
        'axes.edgecolor': '#2a2a2a',
        'axes.labelcolor': '#1a1a1a',
        'text.color': '#1a1a1a',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.color': '#1a1a1a',
        'ytick.color': '#1a1a1a',
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'grid.color': '#d0d0d0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'legend.frameon': False,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'lines.linewidth': 2.0,
    })
    sns.set_style('whitegrid')


class ComparisonDataset(Dataset):
    def __init__(self, comparisons: List[Dict], repr_map: Dict[str, torch.Tensor]):
        self.comparisons = comparisons
        self.repr_map = repr_map

    def __len__(self):
        return len(self.comparisons)

    def __getitem__(self, idx):
        comp = self.comparisons[idx]
        left_id, right_id = comp['left_id'], comp['right_id']
        return {
            'left_repr': self.repr_map[left_id],
            'right_repr': self.repr_map[right_id],
            'label': 1 if comp['winner'] == 'left' else 0,
            'category': comp['category'],
            'left_id': left_id,
            'right_id': right_id,
        }


class BradleyTerryModel(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dims: List[int] = [256, 128, 64], dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.score_predictor = nn.Sequential(*layers)

    def forward(self, graph_repr):
        return self.score_predictor(graph_repr).squeeze()

    def predict_comparison(self, left, right):
        return torch.sigmoid(self.forward(left) - self.forward(right))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if y_prob.size == 0 or len(np.unique(y_true)) < 2:
        return float('nan')
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    total = len(y_prob)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_prob = y_prob[mask]
        bin_true = y_true[mask]
        avg_conf = bin_prob.mean()
        avg_acc = bin_true.mean()
        ece += (np.abs(avg_acc - avg_conf) * mask.sum()) / total
    return float(ece)


def calculate_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if y_prob.size == 0:
        return metrics
    y_pred = (y_prob > 0.5).astype(int)
    unique_labels = np.unique(y_true)
    has_two_classes = len(unique_labels) > 1

    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred) if has_two_classes else float('nan')
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred) if has_two_classes else float('nan')
    metrics['brier'] = brier_score_loss(y_true, y_prob)
    metrics['logloss'] = log_loss(y_true, y_prob_clipped, labels=[0, 1]) if has_two_classes else float('nan')
    metrics['auc'] = roc_auc_score(y_true, y_prob) if has_two_classes else float('nan')
    metrics['ap'] = average_precision_score(y_true, y_prob) if has_two_classes else float('nan')
    metrics['ece'] = expected_calibration_error(y_true, y_prob)
    return metrics


def _load_comparisons_from_file(json_path: str) -> Tuple[List[Dict], Optional[str]]:
    path = Path(json_path)
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return [], f'Failed to read {path} {e}'

    file_id = data.get('file_id')
    if file_id is None:
        return [], None

    comps = []
    for comp in data.get('metadata', {}).get('comparisons', []):
        try:
            role = comp['role']
            opponent_id = comp['opponent_id']
            winner = comp['winner']
            category = comp.get('category', 'unknown')
            if role == 'left':
                left_id, right_id = file_id, opponent_id
            else:
                left_id, right_id = opponent_id, file_id
            comps.append({
                'left_id': left_id,
                'right_id': right_id,
                'winner': winner,
                'category': category,
            })
        except KeyError:
            continue
    return comps, None


def prepare_feature_matrix(comparisons: List[Dict], repr_map: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    left_vecs: List[np.ndarray] = []
    right_vecs: List[np.ndarray] = []
    labels: List[int] = []
    for comp in comparisons:
        left_vecs.append(repr_map[comp['left_id']].cpu().numpy())
        right_vecs.append(repr_map[comp['right_id']].cpu().numpy())
        labels.append(1 if comp['winner'] == 'left' else 0)
    left = np.vstack(left_vecs) if left_vecs else np.zeros((0, 1))
    right = np.vstack(right_vecs) if right_vecs else np.zeros((0, 1))
    diff = left - right
    return diff, np.array(labels)


def evaluate_baselines(comparisons: List[Dict], repr_map: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(42)
    baselines: Dict[str, Dict[str, float]] = {}
    features, labels = prepare_feature_matrix(comparisons, repr_map)
    if labels.size == 0:
        return baselines

    # Random baseline (slightly biased around 0.5)
    random_probs = np.clip(rng.normal(loc=0.5, scale=0.12, size=labels.shape[0]), 1e-6, 1 - 1e-6)
    baselines['random_gaussian'] = calculate_binary_metrics(labels, random_probs)
    baselines['random_gaussian']['loss'] = baselines['random_gaussian']['logloss']

    # Logistic baseline
    try:
        if features.shape[0] > 10:
            subset = min(features.shape[0], 2000)
            log_reg = LogisticRegression(max_iter=200, solver='lbfgs')
            log_reg.fit(features[:subset], labels[:subset])
            probs = log_reg.predict_proba(features)[:, 1]
            baselines['logistic'] = calculate_binary_metrics(labels, probs)
            baselines['logistic']['loss'] = baselines['logistic']['logloss']
    except Exception as exc:
        print(f'Baseline logistic regression failed: {exc}')

    return baselines


def load_graph_representations_with_ids(reps_file: str, ids_file: str) -> Dict[str, torch.Tensor]:
    reps = torch.load(reps_file)
    ids = json.load(open(ids_file, 'r', encoding='utf-8'))
    n = min(len(ids), reps.shape[0])
    return {ids[i]: reps[i] for i in range(n)}


def extract_comparisons_from_graphs(
    graphs_dir: str,
    max_samples: Optional[int] = None,
    workers: int = 0,
) -> List[Dict]:
    p = Path(graphs_dir)
    files = sorted(f for f in p.glob('graph_*.json') if f.is_file())
    if max_samples:
        files = files[:max_samples]
    out = []
    total = len(files)
    if total == 0:
        return out

    if workers is None or workers < 0:
        workers = 0
    if workers == 0:
        worker_count = max(1, os.cpu_count() or 1)
    else:
        worker_count = workers
    worker_count = min(worker_count, total)

    if worker_count <= 1:
        for jf in tqdm(files, desc='Extracting comparisons'):
            comps, err = _load_comparisons_from_file(str(jf))
            if err:
                print(err)
                continue
            out.extend(comps)
    else:
        use_process = os.name != 'nt'
        backend = 'process' if use_process else 'thread'
        print(f'Using parallel extraction: {worker_count} {backend}s')
        chunk_size = max(1, total // (worker_count * 4))
        executor_cls = concurrent.futures.ProcessPoolExecutor if use_process else concurrent.futures.ThreadPoolExecutor
        map_kwargs = {'chunksize': chunk_size} if use_process else {}
        with executor_cls(max_workers=worker_count) as executor:
            results = executor.map(
                _load_comparisons_from_file,
                (str(f) for f in files),
                **map_kwargs,
            )
            for comps, err in tqdm(results, total=total, desc='Extracting comparisons'):
                if err:
                    print(err)
                    continue
                out.extend(comps)
    return out


def filter_comparisons(comparisons: List[Dict], valid_ids: set) -> List[Dict]:
    out = [c for c in comparisons if c['left_id'] in valid_ids and c['right_id'] in valid_ids]
    print(f'Filtered comparison pairs: {len(comparisons)} -> {len(out)}')
    return out


def evaluate(model: nn.Module, device: str, dataloader: DataLoader) -> Dict[str, object]:
    model.eval()
    crit = nn.BCELoss()
    total_loss = 0.0
    n = 0
    preds: List[float] = []
    labels: List[int] = []
    cats: List[str] = []
    left_ids: List[str] = []
    right_ids: List[str] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluate'):
            l = batch['left_repr'].to(device)
            r = batch['right_repr'].to(device)
            y = batch['label'].float().to(device)
            p = model.predict_comparison(l, r)
            total_loss += crit(p, y).item()
            n += y.size(0)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())
            cats.extend(list(batch['category']))
            left_ids.extend(list(batch['left_id']))
            right_ids.extend(list(batch['right_id']))
    y_true = np.array(labels).astype(int)
    y_score = np.array(preds)
    base_metrics = calculate_binary_metrics(y_true, y_score) if labels else {}
    base_metrics['loss'] = total_loss / max(1, n)
    base_metrics['predictions'] = preds
    base_metrics['labels'] = labels
    base_metrics['categories'] = cats
    base_metrics['left_ids'] = left_ids
    base_metrics['right_ids'] = right_ids

    per_cat: Dict[str, Dict[str, float]] = {}
    arr_cat = np.array(cats)
    for c in sorted(set(cats)):
        idx = (arr_cat == c)
        if idx.sum() == 0:
            continue
        cat_metrics = calculate_binary_metrics(y_true[idx], y_score[idx])
        cat_metrics['count'] = int(idx.sum())
        per_cat[c] = cat_metrics
    base_metrics['per_category'] = per_cat
    return base_metrics


def plot_category_auc(per_category: Dict[str, Dict[str, float]], out_path: Path):
    cats = list(per_category.keys())
    aucs = [per_category[c].get('auc', float('nan')) for c in cats]
    counts = [per_category[c].get('count', 0) for c in cats]
    palette = sns.color_palette(NATURE_PALETTE, n_colors=len(cats)) if cats else NATURE_PALETTE
    plt.figure(figsize=(12, 6))
    bars = plt.bar(cats, aucs, color=palette[:len(cats)])
    for b, n in zip(bars, counts):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'n={n}', ha='center', va='bottom', fontsize=9)
    plt.xlabel('Category')
    plt.ylabel('ROC-AUC')
    plt.title('Per-Category ROC-AUC')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=35, ha='right')
    plt.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=350, bbox_inches='tight')
    plt.close()


def plot_roc_pr(labels: List[int], probs: List[float], out_dir: Path):
    y_true = np.array(labels).astype(int)
    y_score = np.array(probs)
    if len(set(y_true)) < 2:
        return
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color=ACCENT_COLOR, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='#6c6c6c', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'roc_curve.png', dpi=350, bbox_inches='tight')
    plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color=NATURE_PALETTE[2], label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / 'pr_curve.png', dpi=350, bbox_inches='tight')
    plt.close()


def plot_confusion(cm: List[List[int]], out_path: Path):
    if cm is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 4.5))
    cmap = sns.color_palette("crest", as_cmap=True)
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap=cmap, cbar=False, linewidths=0.5, linecolor='white')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path, dpi=350, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Bradley-Terry comparison model')
    parser.add_argument('--graphs-dir', type=str, default='output/stage_03_scene_graphs', help='Scene graph JSON directory')
    parser.add_argument('--repr-file', type=str, default='graph_representations.pt', help='Graph representation .pt file path')
    parser.add_argument('--repr-ids', type=str, default='graph_ids.json', help='Graph ID index JSON path')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of scene graph files to read, default all')
    parser.add_argument('--extract-workers', type=int, default=0, help='Number of processes for extracting comparison pairs, 0 means auto')
    parser.add_argument('--batch-size', type=int, default=64, help='Evaluation batch size')
    parser.add_argument('--dataloader-workers', type=int, default=0, help='Number of DataLoader worker threads')
    parser.add_argument('--skip-baselines', action='store_true', help='Skip baseline evaluation to speed up')
    return parser.parse_args()


def main():
    args = parse_args()

    result_dir = Path('result')
    result_dir.mkdir(parents=True, exist_ok=True)
    apply_nature_style()

    # Load representations
    repr_map = load_graph_representations_with_ids(args.repr_file, args.repr_ids)
    if not repr_map:
        print('Graph representations not found, please run 05_graph_vae.py first')
        return

    # Load comparisons
    comps = extract_comparisons_from_graphs(
        args.graphs_dir,
        max_samples=args.max_samples,
        workers=args.extract_workers,
    )
    if not comps:
        print('No comparison pairs found')
        return
    comps = filter_comparisons(comps, set(repr_map.keys()))
    if not comps:
        print('No comparison pairs after filtering')
        return

    # Build dataset
    ds = ComparisonDataset(comps, repr_map)
    dataloader_workers = max(0, args.dataloader_workers)
    if os.name == 'nt' and dataloader_workers > 0:
        print(f'Windows platform temporarily disables DataLoader multiprocessing, automatically adjusting workers={dataloader_workers} to 0')
        dataloader_workers = 0
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BradleyTerryModel(input_dim=128).to(device)
    best_in_result = result_dir / 'bradley_terry_comparison_best.pth'
    best_in_root = Path('bradley_terry_comparison_best.pth')
    ckpt_path = None
    if best_in_result.exists():
        ckpt_path = best_in_result
    elif best_in_root.exists():
        ckpt_path = best_in_root
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'Model loaded: {ckpt_path}')
        # ensure copy into result
        if ckpt_path == best_in_root:
            copy2(ckpt_path, best_in_result)
    else:
        print('Trained model not found, using randomly initialized model for evaluation (testing workflow only).')

    # Evaluate
    metrics = evaluate(model, device, dl)
    baseline_metrics = {}
    if not args.skip_baselines:
        baseline_metrics = evaluate_baselines(comps, repr_map)
    metrics['baselines'] = baseline_metrics

    summary_keys = ['loss', 'accuracy', 'balanced_accuracy', 'auc', 'ap', 'precision', 'recall', 'f1', 'mcc', 'cohen_kappa', 'brier', 'ece']
    print('Overall metrics:', {k: metrics.get(k) for k in summary_keys if k in metrics})
    if baseline_metrics:
        print('Baseline metrics (degraded for comparison):')
        for name, vals in baseline_metrics.items():
            print(f'  {name}: ' + ", ".join(f"{k}={vals.get(k):.4f}" for k in ['accuracy','auc','ap','f1','balanced_accuracy']))

    # Visualize
    plot_category_auc(metrics['per_category'], result_dir / 'category_auc.png')
    plot_roc_pr(metrics['labels'], metrics['predictions'], result_dir)
    y_pred = (np.array(metrics['predictions']) > 0.5).astype(int)
    cm = confusion_matrix(np.array(metrics['labels']).astype(int), y_pred).tolist()
    plot_confusion(cm, result_dir / 'confusion_matrix.png')

    # Save predictions and metrics
    import csv
    with open(result_dir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['left_id','right_id','category','label','prob_left>right','pred_label'])
        for li, ri, cat, y, p in zip(metrics['left_ids'], metrics['right_ids'], metrics['categories'], metrics['labels'], metrics['predictions']):
            w.writerow([li, ri, cat, int(y), float(p), int(p>0.5)])

    with open(result_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in metrics.items() if k not in ['predictions','labels','categories','left_ids','right_ids']}, f, ensure_ascii=False, indent=2)
    print('Visualizations, predictions and metrics saved to ./result')


if __name__ == '__main__':
    main()
