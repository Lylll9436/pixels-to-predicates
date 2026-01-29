#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDP (DistributedDataParallel) Training Script for Dual-Layer Urban Perception Model

This script enables multi-GPU training on 8x V100 GPUs using PyTorch DDP.

Features:
1. Automatic distributed environment detection (torchrun / Slurm / single-GPU fallback)
2. DistributedSampler for data sharding across processes
3. Cross-process metric aggregation (all_reduce, all_gather_object)
4. Rank-0 only model/log saving
5. Linear learning rate scaling (lr * world_size)

Usage:
    # Single GPU (auto-fallback)
    python src/12_dual_layer_trainer_ddp.py --comparisons-file data/comparisons.json

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 src/12_dual_layer_trainer_ddp.py --comparisons-file data/comparisons.json

    # Multi-GPU on Slurm
    srun python src/12_dual_layer_trainer_ddp.py --comparisons-file data/comparisons.json

Author: CEUS Project
Date: 2024
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)
from scipy.stats import spearmanr
from tqdm import tqdm

import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model_dual_layer import DualLayerPerceptionModel, create_dual_layer_model
from micro_encoder import load_all_micro_features, load_caption_features


# ==============================================================================
# Distributed Setup
# ==============================================================================
def setup_distributed() -> Tuple[int, int, int, bool]:
    """
    Initialize distributed training environment.

    Detects:
    1. torchrun environment (RANK, WORLD_SIZE, LOCAL_RANK)
    2. Slurm environment (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID)
    3. Falls back to single-GPU mode

    Returns:
        rank: Global process rank
        world_size: Total number of processes
        local_rank: Local GPU index on this node
        is_distributed: Whether running in distributed mode
    """
    # Check for torchrun environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank, True

    # Check for Slurm environment
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(
            os.environ.get("SLURM_LOCALID", rank % torch.cuda.device_count())
        )

        # Slurm requires explicit MASTER_ADDR and MASTER_PORT
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = os.environ.get(
                "SLURM_LAUNCH_NODE_IPADDR", "localhost"
            )
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank, True

    # Single-GPU fallback
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ==============================================================================
# Logging (Rank-0 Only)
# ==============================================================================
def create_logger(log_dir: Path, name: str, rank: int = 0) -> logging.Logger:
    """Create logger (only rank 0 writes to file and console)."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if rank == 0:
        # File handler
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        # Non-rank-0 processes: null handler
        logger.addHandler(logging.NullHandler())

    return logger


# ==============================================================================
# Dataset (Same as Original)
# ==============================================================================
NEGATIVE_CATEGORIES = {"boring", "depressing"}


class DualLayerComparisonDataset(Dataset):
    """Dual-layer comparison dataset (returns indices, not vectors)."""

    def __init__(
        self,
        comparisons: List[Dict],
        id_to_index: Dict[str, int],
        flip_negative: bool = True,
    ):
        self.comparisons = comparisons
        self.id_to_index = id_to_index
        self.flip_negative = flip_negative

        self.valid_comparisons = []
        for c in comparisons:
            if c["left_id"] in id_to_index and c["right_id"] in id_to_index:
                self.valid_comparisons.append(c)

        filtered_count = len(comparisons) - len(self.valid_comparisons)
        if filtered_count > 0:
            print(f"Filtered {filtered_count} comparisons with missing IDs")

    def __len__(self):
        return len(self.valid_comparisons)

    def __getitem__(self, idx):
        comparison = self.valid_comparisons[idx]
        left_idx = self.id_to_index[comparison["left_id"]]
        right_idx = self.id_to_index[comparison["right_id"]]
        category = comparison.get("category", "")
        winner = comparison.get("winner", "")

        if winner == "left":
            label = 1.0
        elif winner == "right":
            label = 0.0
        elif winner == "equal":
            label = 0.5
        else:
            label = 0.5

        if self.flip_negative and category in NEGATIVE_CATEGORIES:
            label = 1.0 - label

        return {
            "left_idx": left_idx,
            "right_idx": right_idx,
            "label": label,
            "category": comparison["category"],
            "left_id": comparison["left_id"],
            "right_id": comparison["right_id"],
        }


# ==============================================================================
# DDP Trainer
# ==============================================================================
class DualLayerTrainerDDP:
    """Dual-layer perception model trainer with DDP support."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-5,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        is_distributed: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = is_distributed
        self.device = device

        # Move model to device
        model = model.to(device)

        # Wrap with DDP if distributed
        if is_distributed and world_size > 1:
            self.model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        else:
            self.model = model

        # Linear learning rate scaling
        effective_lr = learning_rate * world_size

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=effective_lr, weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )

    def _all_reduce_scalar(self, value: float) -> float:
        """All-reduce a scalar across all processes."""
        if not self.is_distributed:
            return value

        tensor = torch.tensor([value], device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / self.world_size

    def _all_reduce_sum(self, value: float) -> float:
        """All-reduce sum (without averaging) across all processes."""
        if not self.is_distributed:
            return value

        tensor = torch.tensor([value], device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()

    def _all_gather_list(self, local_list: List) -> List:
        """Gather lists from all processes."""
        if not self.is_distributed:
            return local_list

        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_list)
        # Flatten
        return [item for sublist in gathered for item in sublist]

    def train_epoch(
        self,
        dataloader: DataLoader,
        sampler: Optional[DistributedSampler] = None,
        epoch: int = 0,
        grad_clip: float = 1.0,
    ) -> Dict[str, float]:
        self.model.train()

        if sampler is not None:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        correct = 0
        total = 0

        iterator = (
            tqdm(dataloader, desc=f"Training (Rank {self.rank})")
            if self.rank == 0
            else dataloader
        )

        for batch in iterator:
            left_idx = batch["left_idx"].to(self.device)
            right_idx = batch["right_idx"].to(self.device)
            labels = batch["label"].float().to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(left_idx, right_idx)
            loss = self.criterion(predictions, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            predicted_labels = (predictions > 0.5).float()
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

        # Aggregate metrics across processes
        avg_loss = self._all_reduce_scalar(total_loss / len(dataloader))
        total_correct = self._all_reduce_sum(correct)
        total_samples = self._all_reduce_sum(total)

        return {
            "loss": avg_loss,
            "accuracy": total_correct / total_samples if total_samples > 0 else 0,
        }

    def validate(
        self, dataloader: DataLoader, sampler: Optional[DistributedSampler] = None
    ) -> Dict[str, float]:
        """Validate with metric aggregation across processes."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_cats = []

        iterator = (
            tqdm(dataloader, desc=f"Validating (Rank {self.rank})")
            if self.rank == 0
            else dataloader
        )

        with torch.no_grad():
            for batch in iterator:
                left_idx = batch["left_idx"].to(self.device)
                right_idx = batch["right_idx"].to(self.device)
                labels = batch["label"].float().to(self.device)

                predictions = self.model(left_idx, right_idx)
                loss = self.criterion(predictions, labels)

                total_loss += loss.item()

                all_preds.extend(predictions.detach().cpu().view(-1).tolist())
                all_labels.extend(labels.detach().cpu().view(-1).tolist())
                all_cats.extend(batch["category"])

        # Gather predictions from all processes
        all_preds = self._all_gather_list(all_preds)
        all_labels = self._all_gather_list(all_labels)
        all_cats = self._all_gather_list(all_cats)

        # Aggregate loss
        avg_loss = self._all_reduce_scalar(total_loss / len(dataloader))

        # Compute metrics on rank 0 (all processes have same data now)
        metrics = compute_pairwise_metrics(all_preds, all_labels, all_cats)

        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "ap": metrics["ap"],
            "logloss": metrics["logloss"],
            "brier": metrics["brier"],
            "spearman": metrics["spearman"],
            "num_total": metrics["num_total"],
            "num_binary": metrics["num_binary"],
            "num_equal": metrics["num_equal"],
            "predictions": all_preds,
            "labels": all_labels,
            "categories": all_cats,
        }

    def evaluate_full(self, dataloader: DataLoader) -> Dict[str, object]:
        """Full evaluation with per-category metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_cats = []
        all_lids = []
        all_rids = []

        iterator = (
            tqdm(dataloader, desc=f"Evaluating (Rank {self.rank})")
            if self.rank == 0
            else dataloader
        )

        with torch.no_grad():
            for batch in iterator:
                left_idx = batch["left_idx"].to(self.device)
                right_idx = batch["right_idx"].to(self.device)
                labels = batch["label"].float().to(self.device)

                predictions = self.model(left_idx, right_idx)
                predictions = predictions.view(-1)
                labels = labels.view(-1)

                all_preds.extend(predictions.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())
                all_cats.extend(batch["category"])
                all_lids.extend(batch["left_id"])
                all_rids.extend(batch["right_id"])

        # Gather from all processes
        all_preds = self._all_gather_list(all_preds)
        all_labels = self._all_gather_list(all_labels)
        all_cats = self._all_gather_list(all_cats)
        all_lids = self._all_gather_list(all_lids)
        all_rids = self._all_gather_list(all_rids)

        metrics = compute_pairwise_metrics(all_preds, all_labels, all_cats)
        metrics["predictions"] = all_preds
        metrics["labels"] = all_labels
        metrics["categories"] = all_cats
        metrics["left_ids"] = all_lids
        metrics["right_ids"] = all_rids

        return metrics


# ==============================================================================
# Metrics Computation (Same as Original)
# ==============================================================================
def compute_pairwise_metrics(
    predictions: List[float], labels: List[float], categories: List[str]
) -> Dict[str, object]:
    """Compute pairwise comparison metrics."""
    if len(predictions) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": float("nan"),
            "ap": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
            "spearman": float("nan"),
            "per_category": {},
            "num_total": 0,
            "num_binary": 0,
            "num_equal": 0,
        }

    y_prob = np.asarray(predictions, dtype=np.float32)
    y_true = np.asarray(labels, dtype=np.float32)
    cat_arr = np.asarray(categories)

    def _calc_metrics(
        y_prob_sub: np.ndarray, y_true_sub: np.ndarray
    ) -> Dict[str, float]:
        metrics_sub: Dict[str, float] = {}
        binary_mask = (y_true_sub == 0.0) | (y_true_sub == 1.0)
        num_binary = int(binary_mask.sum())

        if num_binary > 0:
            y_true_bin = y_true_sub[binary_mask]
            y_prob_bin = y_prob_sub[binary_mask]
            y_cls = (y_prob_bin > 0.5).astype(int)

            metrics_sub["accuracy"] = float(accuracy_score(y_true_bin, y_cls))
            metrics_sub["precision"] = float(
                precision_score(y_true_bin, y_cls, zero_division=0)
            )
            metrics_sub["recall"] = float(
                recall_score(y_true_bin, y_cls, zero_division=0)
            )
            metrics_sub["f1"] = float(f1_score(y_true_bin, y_cls, zero_division=0))

            try:
                metrics_sub["auc"] = (
                    float(roc_auc_score(y_true_bin, y_prob_bin))
                    if len(np.unique(y_true_bin)) > 1
                    else float("nan")
                )
            except Exception:
                metrics_sub["auc"] = float("nan")

            try:
                metrics_sub["ap"] = (
                    float(average_precision_score(y_true_bin, y_prob_bin))
                    if len(np.unique(y_true_bin)) > 1
                    else float("nan")
                )
            except Exception:
                metrics_sub["ap"] = float("nan")

            try:
                metrics_sub["logloss"] = float(
                    log_loss(y_true_bin, np.clip(y_prob_bin, 1e-7, 1 - 1e-7))
                )
            except Exception:
                metrics_sub["logloss"] = float("nan")

            try:
                metrics_sub["brier"] = float(brier_score_loss(y_true_bin, y_prob_bin))
            except Exception:
                metrics_sub["brier"] = float("nan")
        else:
            metrics_sub["accuracy"] = float("nan")
            metrics_sub["precision"] = float("nan")
            metrics_sub["recall"] = float("nan")
            metrics_sub["f1"] = float("nan")
            metrics_sub["auc"] = float("nan")
            metrics_sub["ap"] = float("nan")
            metrics_sub["logloss"] = float("nan")
            metrics_sub["brier"] = float("nan")

        try:
            metrics_sub["spearman"] = float(
                spearmanr(y_true_sub, y_prob_sub).correlation
            )
        except Exception:
            metrics_sub["spearman"] = float("nan")

        metrics_sub["num_total"] = int(len(y_true_sub))
        metrics_sub["num_binary"] = num_binary
        metrics_sub["num_equal"] = int((y_true_sub == 0.5).sum())
        return metrics_sub

    metrics = _calc_metrics(y_prob, y_true)

    # Per-category metrics
    per_category: Dict[str, Dict[str, float]] = {}
    for category in sorted(set(cat_arr.tolist())):
        mask = cat_arr == category
        if mask.sum() == 0:
            continue
        per_category[category] = _calc_metrics(y_prob[mask], y_true[mask])
        per_category[category]["count"] = int(mask.sum())
    metrics["per_category"] = per_category

    return metrics


# ==============================================================================
# Data Loading (Same as Original)
# ==============================================================================
def load_comparisons_from_file(
    comparisons_file: str, max_samples: Optional[int] = None, rank: int = 0
) -> List[Dict]:
    """Load comparisons from file (JSON or CSV format)."""
    import csv

    if rank == 0:
        print(f"Loading comparisons from: {comparisons_file}")

    file_path = Path(comparisons_file)

    if file_path.suffix.lower() == ".csv":
        comparisons = []
        with open(comparisons_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                comparisons.append(
                    {
                        "left_id": row["left_id"],
                        "right_id": row["right_id"],
                        "winner": row["winner"],
                        "category": row["category"],
                    }
                )
    else:
        with open(comparisons_file, "r", encoding="utf-8") as f:
            comparisons = json.load(f)

    if max_samples:
        comparisons = comparisons[:max_samples]

    if rank == 0:
        print(f"Loaded {len(comparisons)} comparisons")

        category_counts = {}
        for comp in comparisons:
            category = comp["category"]
            category_counts[category] = category_counts.get(category, 0) + 1

        print("Category distribution:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")

    return comparisons


def extract_comparisons_from_graphs(
    graphs_dir: str, max_samples: Optional[int] = None, rank: int = 0
) -> List[Dict]:
    """Extract comparisons from scene graph files."""
    graphs_path = Path(graphs_dir)
    json_files = list(graphs_path.glob("graph_*.json"))

    if max_samples:
        json_files = json_files[:max_samples]

    comparisons = []
    iterator = (
        tqdm(json_files, desc="Extracting comparisons") if rank == 0 else json_files
    )

    for json_file in iterator:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_id = data["file_id"]
            comparisons_data = data.get("metadata", {}).get("comparisons", [])

            for comp in comparisons_data:
                comparison = {
                    "left_id": file_id
                    if comp["role"] == "left"
                    else comp["opponent_id"],
                    "right_id": comp["opponent_id"]
                    if comp["role"] == "left"
                    else file_id,
                    "winner": comp["winner"],
                    "category": comp["category"],
                }
                comparisons.append(comparison)

        except Exception as e:
            if rank == 0:
                print(f"Error processing {json_file}: {e}")
            continue

    if rank == 0:
        print(f"Extracted {len(comparisons)} comparisons")

        category_counts = {}
        for comp in comparisons:
            category = comp["category"]
            category_counts[category] = category_counts.get(category, 0) + 1

        print("Category distribution:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")

    return comparisons


def sanitize_float(val) -> float:
    """Sanitize float values (handle NaN/Inf)."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except:
        return None


def save_metrics_json(json_path: Path, metrics: Dict, extra: Optional[Dict] = None):
    """Save metrics to JSON file."""
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        k: v
        for k, v in metrics.items()
        if k not in ["predictions", "labels", "categories", "left_ids", "right_ids"]
    }

    for k, v in payload.items():
        if isinstance(v, float):
            payload[k] = sanitize_float(v)

    if extra:
        payload.update(extra)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ==============================================================================
# HPO Objective Function
# ==============================================================================
def create_objective(
    args,
    h_visual,
    h_caption,
    visual_dim,
    caption_dim,
    macro_graph,
    image_to_street_tensor,
    train_ds,
    val_ds,
    device,
    rank,
    world_size,
    logger,
):
    def objective(trial: Trial) -> float:
        gnn_type = trial.suggest_categorical(
            "gnn_type", ["GAT", "GATv2", "SAGE", "GCN", "Transformer", "GIN"]
        )
        num_gnn_layers = trial.suggest_int("num_gnn_layers", 1, 4)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        gnn_aggr = trial.suggest_categorical("gnn_aggr", ["mean", "max", "sum"])

        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        micro_dim = trial.suggest_categorical("micro_dim", [64, 128, 256])

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        grad_clip = trial.suggest_float("grad_clip", 0.5, 5.0)

        lr_factor = trial.suggest_float("lr_factor", 0.3, 0.7)
        lr_patience = trial.suggest_int("lr_patience", 3, 10)

        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)

        if rank == 0:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Trial {trial.number}: Testing hyperparameters")
            logger.info(
                f"  GNN: {gnn_type}, Layers: {num_gnn_layers}, Heads: {num_heads}"
            )
            logger.info(f"  Hidden: {hidden_dim}, Micro: {micro_dim}")
            logger.info(
                f"  LR: {learning_rate:.2e}, WD: {weight_decay:.2e}, Dropout: {dropout:.2f}"
            )
            logger.info(f"{'=' * 60}\n")

        model = create_dual_layer_model(
            h_visual=h_visual,
            h_caption=h_caption,
            visual_dim=visual_dim,
            caption_dim=caption_dim,
            micro_fusion_mode=args.fusion_mode,
            micro_feature_dim=micro_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout,
            gnn_type=gnn_type,
            gnn_aggr=gnn_aggr,
            macro_graph_data=macro_graph,
            image_to_street_mapping=image_to_street_tensor,
        )

        model = model.to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[rank])

        if world_size > 1:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True
            )
            val_sampler = DistributedSampler(
                val_ds, num_replicas=world_size, rank=rank, shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=0,
        )

        effective_lr = learning_rate * world_size
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=effective_lr, weight_decay=weight_decay
        )
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=lr_factor, patience=lr_patience, min_lr=1e-6
        )

        best_auc = 0.0

        for epoch in range(1, args.hpo_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            for batch in train_loader:
                left_idx = batch["left_idx"].to(device)
                right_idx = batch["right_idx"].to(device)
                labels = batch["label"].float().to(device)

                optimizer.zero_grad()
                predictions = model(left_idx, right_idx)
                loss = criterion(predictions, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    left_idx = batch["left_idx"].to(device)
                    right_idx = batch["right_idx"].to(device)
                    labels = batch["label"].float().to(device)
                    predictions = model(left_idx, right_idx)
                    all_preds.extend(predictions.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            y_prob = np.array(all_preds)
            y_true = np.array(all_labels)
            binary_mask = (y_true == 0.0) | (y_true == 1.0)
            if binary_mask.sum() > 0:
                y_true_bin = y_true[binary_mask]
                y_prob_bin = y_prob[binary_mask]
                try:
                    val_auc = (
                        roc_auc_score(y_true_bin, y_prob_bin)
                        if len(np.unique(y_true_bin)) > 1
                        else 0.5
                    )
                except:
                    val_auc = 0.5
            else:
                val_auc = 0.5

            scheduler.step(val_auc)

            if rank == 0:
                logger.info(
                    f"Trial {trial.number} Epoch {epoch}/{args.hpo_epochs}: Val AUC {val_auc:.4f}"
                )

            if val_auc > best_auc:
                best_auc = val_auc

            trial.report(val_auc, epoch)

            if trial.should_prune():
                if rank == 0:
                    logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()

        return best_auc

    return objective


# ==============================================================================
# Main Training Flow (DDP Version)
# ==============================================================================
def run_training(args) -> None:
    """Run DDP training."""

    # ====== Setup Distributed ======
    rank, world_size, local_rank, is_distributed = setup_distributed()

    # Device
    if is_distributed:
        device = f"cuda:{local_rank}"
    else:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Directories
    result_dir = Path(args.result_dir)
    if rank == 0:
        result_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / "dual_layer_ddp"
    logger = create_logger(log_dir, f"dual_layer_{args.fusion_mode}_ddp", rank=rank)

    logger.info("=" * 60)
    logger.info("DDP Training: Dual-Layer Urban Perception Model")
    logger.info("=" * 60)
    logger.info(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    logger.info(f"Device: {device}")
    logger.info(f"Distributed: {is_distributed}")
    logger.info(f"Fusion Mode: {args.fusion_mode}")
    logger.info(f"Use Functional Edges: {not args.no_func_edges}")

    # ====== 1. Load Micro Features ======
    logger.info("\n[Step 1] Loading micro features...")

    repr_file = Path(args.repr_file)
    ids_file = Path(args.ids_file)
    graphs_dir = Path(args.graphs_dir)

    if not repr_file.exists() or not ids_file.exists():
        logger.error(f"Micro feature files not found: {repr_file} or {ids_file}")
        cleanup_distributed()
        return

    # Load visual features
    h_visual = torch.load(repr_file, map_location="cpu", weights_only=False)
    with open(ids_file, "r", encoding="utf-8") as f:
        graph_ids = json.load(f)

    visual_dim = h_visual.shape[1]
    logger.info(f"Visual features: {h_visual.shape[0]} samples, dim {visual_dim}")

    # Load caption features
    h_caption = load_caption_features(str(graphs_dir), graph_ids)
    caption_dim = h_caption.shape[1]
    logger.info(f"Caption features: {h_caption.shape[0]} samples, dim {caption_dim}")

    id_to_index = {fid: i for i, fid in enumerate(graph_ids)}

    # ====== 2. Load Macro Graph ======
    logger.info("\n[Step 2] Loading macro graph...")

    macro_graph_path = Path(args.macro_graph)
    macro_map_path = Path(args.macro_mapping)

    if not macro_graph_path.exists() or not macro_map_path.exists():
        logger.error(
            f"Macro graph files not found: {macro_graph_path} or {macro_map_path}"
        )
        cleanup_distributed()
        return

    macro_graph = torch.load(macro_graph_path, map_location="cpu", weights_only=False)
    with open(macro_map_path, "rb") as f:
        img_to_street_dict = pickle.load(f)

    # Ablation: remove functional edges
    if args.no_func_edges and hasattr(macro_graph, "edge_index_func"):
        logger.info("Ablation: Removing functional edges (w/o Macro_Func)")
        macro_graph.edge_index_func = None

    logger.info(f"Macro graph: {macro_graph.num_nodes} street nodes")

    # Build image-to-street mapping tensor
    num_images = len(graph_ids)
    image_to_street_tensor = torch.zeros(num_images, dtype=torch.long)

    missing_streets = 0
    for i, fid in enumerate(graph_ids):
        if fid in img_to_street_dict:
            image_to_street_tensor[i] = img_to_street_dict[fid]
        else:
            missing_streets += 1

    if missing_streets > 0:
        logger.warning(
            f"{missing_streets} images missing street mapping, assigned to street 0"
        )

    # ====== 3. Prepare Dataset ======
    logger.info("\n[Step 3] Preparing dataset...")

    # DDP requires comparisons-file
    if not args.comparisons_file:
        logger.error("--comparisons-file is required for DDP training")
        cleanup_distributed()
        return

    comparisons = load_comparisons_from_file(
        args.comparisons_file, max_samples=args.max_samples, rank=rank
    )

    if args.category:
        allowed = {c.strip() for c in args.category.split(",") if c.strip()}
        comparisons = [c for c in comparisons if c.get("category") in allowed]
        logger.info(f"Category filter: {sorted(allowed)} -> {len(comparisons)} samples")

    if not comparisons:
        logger.error("No comparison data found")
        cleanup_distributed()
        return

    flip_negative = not getattr(args, "no_flip_negative", False)
    if flip_negative:
        logger.info(f"Flipping negative category labels: {NEGATIVE_CATEGORIES}")

    dataset = DualLayerComparisonDataset(
        comparisons, id_to_index, flip_negative=flip_negative
    )

    if len(dataset) < 10:
        logger.error(f"Too few valid samples: {len(dataset)}")
        cleanup_distributed()
        return

    # Split dataset (use same seed across all processes for determinism)
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    logger.info(
        f"Dataset split: Train {len(train_ds)}, Val {len(val_ds)}, Test {len(test_ds)}"
    )

    # ====== Create DataLoaders with DistributedSampler ======
    workers = 0 if os.name == "nt" else args.num_workers

    # Train sampler and loader
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if is_distributed
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers,
        drop_last=True,  # Important for DDP to avoid size mismatch
        pin_memory=True,
    )

    # Validation sampler and loader
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if is_distributed
        else None
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=workers,
        pin_memory=True,
    )

    # Test loader (no sampler needed for final eval - will gather from all)
    test_sampler = (
        DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if is_distributed and len(test_ds) > 0
        else None
    )

    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=workers,
            pin_memory=True,
        )
        if len(test_ds) > 0
        else None
    )

    # Full dataset loader for final evaluation
    full_sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if is_distributed
        else None
    )

    full_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=full_sampler,
        num_workers=workers,
        pin_memory=True,
    )

    # ====== 4. Initialize Model ======
    logger.info("\n[Step 4] Initializing model...")

    model = create_dual_layer_model(
        h_visual=h_visual,
        h_caption=h_caption,
        visual_dim=visual_dim,
        caption_dim=caption_dim,
        micro_fusion_mode=args.fusion_mode,
        micro_feature_dim=args.micro_dim,
        macro_graph_data=macro_graph,
        image_to_street_mapping=image_to_street_tensor,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        gnn_aggr=args.gnn_aggr,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: Total {total_params:,}, Trainable {trainable_params:,}"
    )

    # ====== 5. Training ======
    logger.info("\n[Step 5] Starting training...")
    logger.info(
        f"Effective learning rate: {args.learning_rate * world_size:.2e} (base {args.learning_rate:.2e} x {world_size})"
    )

    trainer = DualLayerTrainerDDP(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=is_distributed,
    )

    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=1e-6,
    )

    best_auc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Synchronize before epoch
        if is_distributed:
            dist.barrier()

        train_res = trainer.train_epoch(
            train_loader, sampler=train_sampler, epoch=epoch, grad_clip=args.grad_clip
        )
        val_res = trainer.validate(val_loader, sampler=val_sampler)

        # Learning rate scheduling
        trainer.scheduler.step(val_res["auc"])
        current_lr = trainer.optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_res["loss"],
                "train_acc": train_res["accuracy"],
                "val_loss": val_res["loss"],
                "val_acc": val_res["accuracy"],
                "val_auc": val_res["auc"],
                "lr": current_lr,
            }
        )

        logger.info(
            f"Epoch {epoch}/{args.epochs}: "
            f"Train Loss {train_res['loss']:.4f} Acc {train_res['accuracy']:.4f} | "
            f"Val AUC {val_res['auc']:.4f} Acc {val_res['accuracy']:.4f} | "
            f"LR {current_lr:.2e}"
        )

        # Save best model (rank 0 only)
        if val_res["auc"] > best_auc:
            best_auc = val_res["auc"]
            # Get underlying model (unwrap DDP)
            model_to_save = (
                trainer.model.module
                if hasattr(trainer.model, "module")
                else trainer.model
            )
            best_state = {
                k: v.cpu().clone() for k, v in model_to_save.state_dict().items()
            }
            patience_counter = 0
            logger.info(f"  New best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered (patience={args.patience})")
                break

    # ====== 6. Evaluation ======
    logger.info("\n[Step 6] Final evaluation...")

    # Synchronize before evaluation
    if is_distributed:
        dist.barrier()

    # Load best model
    if best_state is not None:
        model_to_load = (
            trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        )
        model_to_load.load_state_dict(best_state)

    # Test set evaluation
    if test_loader is not None:
        test_metrics = trainer.evaluate_full(test_loader)
        logger.info(
            f"Test set - AUC: {test_metrics['auc']:.4f}, Acc: {test_metrics['accuracy']:.4f}"
        )
    else:
        test_metrics = {
            "auc": float("nan"),
            "accuracy": float("nan"),
            "spearman": float("nan"),
        }
        logger.info("Test set is empty, skipping evaluation")

    # Full dataset evaluation
    full_metrics = trainer.evaluate_full(full_loader)

    # ====== 7. Save Results (Rank 0 Only) ======
    if rank == 0:
        logger.info("\n[Step 7] Saving results...")

        model_name = f"dual_layer_{args.fusion_mode}_ddp"
        if args.no_func_edges:
            model_name += "_no_func"

        model_path = result_dir / f"{model_name}_best.pth"
        torch.save(best_state, model_path)
        logger.info(f"Model saved: {model_path}")

        extra = {
            "config": {
                "fusion_mode": args.fusion_mode,
                "no_func_edges": args.no_func_edges,
                "micro_dim": args.micro_dim,
                "hidden_dim": args.hidden_dim,
                "num_gnn_layers": args.num_gnn_layers,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "effective_learning_rate": args.learning_rate * world_size,
                "world_size": world_size,
                "category": args.category,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
            },
            "best_val_auc": sanitize_float(best_auc),
            "test_auc": sanitize_float(test_metrics["auc"]),
            "test_accuracy": sanitize_float(test_metrics["accuracy"]),
            "test_spearman": sanitize_float(test_metrics.get("spearman", float("nan"))),
            "history": history,
        }

        metrics_path = result_dir / f"metrics_{model_name}.json"
        save_metrics_json(metrics_path, full_metrics, extra=extra)
        logger.info(f"Metrics saved: {metrics_path}")

    # ====== Cleanup ======
    cleanup_distributed()

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation AUC: {best_auc:.4f}")
    logger.info(f"Test set AUC: {test_metrics['auc']:.4f}")
    logger.info("=" * 60)


# ==============================================================================
# HPO Mode
# ==============================================================================
def run_hpo(args):
    rank, world_size, local_rank, is_distributed = setup_distributed()

    if is_distributed:
        device = f"cuda:{local_rank}"
    else:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    result_dir = Path(args.result_dir)
    if rank == 0:
        result_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / "hpo"
    logger = create_logger(log_dir, f"hpo_rank{rank}", rank=rank)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Bayesian Hyperparameter Optimization")
        logger.info("=" * 60)
        logger.info(f"Study: {args.study_name}")
        logger.info(f"Trials: {args.n_trials}")
        logger.info(f"Epochs per trial: {args.hpo_epochs}")

    repr_file = Path(args.repr_file)
    ids_file = Path(args.ids_file)
    graphs_dir = Path(args.graphs_dir)

    if not repr_file.exists() or not ids_file.exists():
        logger.error(f"Micro feature files not found: {repr_file} or {ids_file}")
        cleanup_distributed()
        return

    h_visual = torch.load(repr_file, map_location="cpu", weights_only=False)
    with open(ids_file, "r", encoding="utf-8") as f:
        graph_ids = json.load(f)

    visual_dim = h_visual.shape[1]
    h_caption = load_caption_features(str(graphs_dir), graph_ids)
    caption_dim = h_caption.shape[1]
    id_to_index = {fid: i for i, fid in enumerate(graph_ids)}

    macro_graph_path = Path(args.macro_graph)
    macro_map_path = Path(args.macro_mapping)

    if not macro_graph_path.exists() or not macro_map_path.exists():
        logger.error(
            f"Macro graph files not found: {macro_graph_path} or {macro_map_path}"
        )
        cleanup_distributed()
        return

    macro_graph = torch.load(macro_graph_path, map_location="cpu", weights_only=False)
    with open(macro_map_path, "rb") as f:
        img_to_street_dict = pickle.load(f)

    if args.no_func_edges and hasattr(macro_graph, "edge_index_func"):
        macro_graph.edge_index_func = None

    num_images = len(graph_ids)
    image_to_street_tensor = torch.zeros(num_images, dtype=torch.long)
    for i, fid in enumerate(graph_ids):
        if fid in img_to_street_dict:
            image_to_street_tensor[i] = img_to_street_dict[fid]

    if not args.comparisons_file:
        logger.error("--comparisons-file is required for HPO")
        cleanup_distributed()
        return

    comparisons = load_comparisons_from_file(
        args.comparisons_file, max_samples=args.max_samples, rank=rank
    )

    if args.category:
        allowed = {c.strip() for c in args.category.split(",") if c.strip()}
        comparisons = [c for c in comparisons if c.get("category") in allowed]

    if not comparisons:
        logger.error("No comparison data found")
        cleanup_distributed()
        return

    flip_negative = not getattr(args, "no_flip_negative", False)
    dataset = DualLayerComparisonDataset(
        comparisons, id_to_index, flip_negative=flip_negative
    )

    if len(dataset) < 10:
        logger.error(f"Too few valid samples: {len(dataset)}")
        cleanup_distributed()
        return

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, _ = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    if rank == 0:
        logger.info(f"Dataset split: Train {len(train_ds)}, Val {len(val_ds)}")

    objective = create_objective(
        args,
        h_visual,
        h_caption,
        visual_dim,
        caption_dim,
        macro_graph,
        image_to_street_tensor,
        train_ds,
        val_ds,
        device,
        rank,
        world_size,
        logger,
    )

    if rank == 0:
        storage = args.storage or f"sqlite:///{result_dir}/optuna_{args.study_name}.db"

        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=TPESampler(seed=args.seed),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )

        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        logger.info("\n" + "=" * 60)
        logger.info("Optimization Complete!")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best AUC: {study.best_trial.value:.4f}")
        logger.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        best_params_path = result_dir / f"best_params_{args.study_name}.json"
        with open(best_params_path, "w") as f:
            json.dump(study.best_trial.params, f, indent=2)
        logger.info(f"Best params saved to: {best_params_path}")

    cleanup_distributed()


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="DDP Training for Dual-Layer Urban Perception Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--repr-file",
        type=str,
        default="output/graph_representations.pt",
        help="GraphMAE visual feature file",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default="output/graph_ids.json",
        help="Graph ID mapping file",
    )
    parser.add_argument(
        "--graphs-dir",
        type=str,
        default="output/scene_graphs",
        help="Scene graph directory",
    )
    parser.add_argument(
        "--macro-graph",
        type=str,
        default="output/macro_graph.pt",
        help="Macro graph file",
    )
    parser.add_argument(
        "--macro-mapping",
        type=str,
        default="output/macro_graph.mapping.pkl",
        help="Image to street mapping file",
    )

    # Comparison data (REQUIRED for DDP)
    parser.add_argument(
        "--comparisons-file",
        type=str,
        required=True,
        help="Comparison data JSON file (required for DDP)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Train only on specified categories (comma-separated)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation set ratio"
    )
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")

    # Output paths
    parser.add_argument(
        "--result-dir", type=str, default="result", help="Result save directory"
    )
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")

    # Model configuration
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="gated",
        choices=["gated", "concat", "visual_only", "caption_only"],
        help="Micro-level fusion mode",
    )
    parser.add_argument(
        "--no-func-edges",
        action="store_true",
        help="Ablation: remove functional edges (w/o Macro_Func)",
    )
    parser.add_argument(
        "--micro-dim",
        type=int,
        default=128,
        help="Micro feature dimension after fusion",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--num-gnn-layers", type=int, default=2, help="Number of GAT layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout ratio")

    # Training configuration
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Base learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=5e-5, help="Weight decay")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker count"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples (for debugging)"
    )

    # Device (for single-GPU fallback)
    parser.add_argument(
        "--device", type=str, default=None, help="Compute device (cuda/cpu)"
    )

    # DDP-specific (auto-detected, but can be overridden)
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for DDP (auto-detected)"
    )

    parser.add_argument(
        "--no-flip-negative",
        action="store_true",
        help="Do not flip negative category labels (boring, depressing)",
    )

    # HPO 参数
    parser.add_argument("--hpo", action="store_true", help="启用 Optuna 超参数搜索")
    parser.add_argument("--n-trials", type=int, default=100, help="HPO trials 数量")
    parser.add_argument(
        "--study-name", type=str, default="dual_layer_hpo", help="Optuna study 名称"
    )
    parser.add_argument(
        "--storage", type=str, default=None, help="Optuna 存储路径 (SQLite)"
    )
    parser.add_argument(
        "--hpo-epochs", type=int, default=20, help="每个 trial 的训练轮数"
    )

    # GNN 架构参数
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="GAT",
        choices=["GAT", "GATv2", "SAGE", "GCN", "Transformer", "GIN"],
        help="GNN 架构类型",
    )
    parser.add_argument(
        "--gnn-aggr",
        type=str,
        default="mean",
        choices=["mean", "max", "sum"],
        help="SAGE 聚合方式",
    )

    # 额外训练参数
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪值")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="标签平滑")
    parser.add_argument(
        "--lr-factor", type=float, default=0.5, help="LR 调度器衰减因子"
    )
    parser.add_argument("--lr-patience", type=int, default=5, help="LR 调度器耐心值")

    args = parser.parse_args()

    if args.hpo:
        run_hpo(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
