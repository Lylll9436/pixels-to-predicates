#!/usr/bin/env python3
"""
Comparison-based trainer - Specifically handles pairwise comparison data
Uses Bradley-Terry model to learn latent scores of graphs
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from tqdm import tqdm
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
)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torchvision
    from torchvision import transforms
    from torchvision.models import (
        ResNet50_Weights,
        ViT_B_16_Weights,
        resnet50,
        vit_b_16,
    )
except Exception:
    torchvision = None
    transforms = None

try:
    import open_clip
except Exception:
    open_clip = None


class ComparisonDataset(Dataset):
    """Comparison dataset"""
    
    def __init__(self, comparisons: List[Dict], graph_representations: Dict[str, torch.Tensor]):
        self.comparisons = comparisons
        self.graph_representations = graph_representations
    
    def __len__(self):
        return len(self.comparisons)
    
    def __getitem__(self, idx):
        comparison = self.comparisons[idx]
        
        # Get representation vectors of two graphs
        left_id = comparison['left_id']
        right_id = comparison['right_id']
        
        left_repr = self.graph_representations[left_id]
        right_repr = self.graph_representations[right_id]
        
        # Label: 1 means left is better, 0 means right is better
        label = 1 if comparison['winner'] == 'left' else 0
        
        return {
            'left_repr': left_repr,
            'right_repr': right_repr,
            'label': label,
            'category': comparison['category'],
            'left_id': left_id,
            'right_id': right_id
        }


@dataclass
class EndToEndConfig:
    backbone: str
    image_root: Path
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 5e-5
    patience: int = 20
    num_workers: int = 4
    split_seed: int = 42
    freeze_backbone: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    result_dir: Path = Path('result')
    log_dir: Path = Path('logs')
    max_samples: Optional[int] = None


def create_logger(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_dir / f'{name}.log', encoding='utf-8')
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_image_index(image_root: Path) -> Dict[str, Path]:
    """Scan image directory, build file_id -> image path mapping"""
    exts = ['.jpg', '.jpeg', '.png', '.webp']
    mapping: Dict[str, Path] = {}
    if not image_root.exists():
        raise FileNotFoundError(f'Image root not found: {image_root}')
    for ext in exts:
        for file_path in image_root.glob(f'*{ext}'):
            mapping[file_path.stem] = file_path
    return mapping


class ImageComparisonDataset(Dataset):
    """Comparison dataset that directly reads images"""
    
    def __init__(
        self,
        comparisons: List[Dict],
        image_index: Dict[str, Path],
        transform,
    ):
        self.transform = transform
        self.samples: List[Dict] = []
        missing = 0
        for item in comparisons:
            left_id = item['left_id']
            right_id = item['right_id']
            if left_id not in image_index or right_id not in image_index:
                missing += 1
                continue
            self.samples.append({
                'left_path': image_index[left_id],
                'right_path': image_index[right_id],
                'left_id': left_id,
                'right_id': right_id,
                'label': 1 if item['winner'] == 'left' else 0,
                'category': item['category'],
            })
        if missing:
            print(f'⚠ Removed {missing} comparisons missing images')
        print(f'✔ ImageComparisonDataset: {len(self.samples)} valid pairs')
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        left_img = Image.open(sample['left_path']).convert('RGB')
        right_img = Image.open(sample['right_path']).convert('RGB')
        if self.transform is not None:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        return {
            'left_image': left_img,
            'right_image': right_img,
            'label': float(sample['label']),
            'category': sample['category'],
            'left_id': sample['left_id'],
            'right_id': sample['right_id'],
        }


class CLIPVisualWrapper(nn.Module):
    """Wrap CLIP model's vision branch as a general feature extractor"""
    
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(images)


def build_feature_extractor(backbone: str, device: str):
    """Build feature extractor and preprocessing based on different backbones"""
    backbone = backbone.lower()
    if backbone == 'cnn':
        if torchvision is None:
            raise ImportError("torchvision is not installed in current environment, cannot use CNN baseline.")
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        transform = weights.transforms()
        model_name = 'resnet50_imagenet'
    elif backbone == 'vit':
        if torchvision is not None and hasattr(torchvision.models, 'vit_b_16'):
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            model = vit_b_16(weights=weights)
            feature_dim = model.heads.head.in_features
            model.heads.head = nn.Identity()
            transform = weights.transforms()
            model_name = 'vit_b16_imagenet'
        else:
            try:  # pragma: no cover - 可选依赖
                import timm  # type: ignore
            except Exception as exc:
                raise ImportError("torchvision>=0.13 or timm is required to use ViT baseline.") from exc
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            feature_dim = model.get_classifier().in_features  # type: ignore[attr-defined]
            model.reset_classifier(0)
            transform = timm.data.create_transform(
                input_size=(3, 224, 224),
                crop_pct=0.9,
                interpolation='bicubic',
            )
            model_name = 'vit_base_patch16_224'
    elif backbone == 'clip':
        if open_clip is None:
            raise ImportError("open_clip is not installed in current environment, cannot use CLIP baseline.")
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k'
        )
        clip_model.visual.train()
        feature_dim = clip_model.visual.output_dim
        model = CLIPVisualWrapper(clip_model)
        transform = preprocess
        model_name = 'clip_vit_b32'
    else:
        raise ValueError(f'Unknown backbone: {backbone}')
    
    model = model.to(device)
    return model, feature_dim, transform, model_name


class EndToEndComparisonModel(nn.Module):
    """End-to-end comparison model (feature extraction + Bradley-Terry score head)"""
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad_(False)
            self.feature_extractor.eval()
        
        layers: List[nn.Module] = []
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.score_head = nn.Sequential(*layers)
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.feature_extractor.eval()
        return self
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        grad_state = not self.freeze_backbone
        with torch.set_grad_enabled(grad_state):
            features = self.feature_extractor(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        if self.freeze_backbone:
            features = features.detach()
        return features
    
    def score(self, features: torch.Tensor) -> torch.Tensor:
        return self.score_head(features).squeeze(-1)
    
    def forward(self, left_images: torch.Tensor, right_images: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        left_features = self.encode(left_images)
        right_features = self.encode(right_images)
        left_score = self.score(left_features)
        right_score = self.score(right_features)
        logits = left_score - right_score
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
    
    def predict(self, left_images: torch.Tensor, right_images: torch.Tensor) -> torch.Tensor:
        return self.forward(left_images, right_images, apply_sigmoid=True)


def compute_pairwise_metrics(predictions: List[float], labels: List[float], categories: List[str]) -> Dict[str, object]:
    """Calculate overall and per-dimension binary classification metrics from probability output"""
    if len(predictions) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': float('nan'),
            'ap': float('nan'),
            'logloss': float('nan'),
            'brier': float('nan'),
            'per_category': {},
        }
    
    y_prob = np.asarray(predictions, dtype=np.float32)
    y_true = np.asarray(labels, dtype=np.float32)
    y_cls = (y_prob > 0.5).astype(int)
    cat_arr = np.asarray(categories)
    
    metrics: Dict[str, object] = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_cls))
    metrics['precision'] = float(precision_score(y_true, y_cls, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_cls, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_cls, zero_division=0))
    try:
        metrics['auc'] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan')
    except Exception:
        metrics['auc'] = float('nan')
    try:
        metrics['ap'] = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan')
    except Exception:
        metrics['ap'] = float('nan')
    try:
        metrics['logloss'] = float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)))
    except Exception:
        metrics['logloss'] = float('nan')
    try:
        metrics['brier'] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        metrics['brier'] = float('nan')
    
    per_category: Dict[str, Dict[str, float]] = {}
    for category in sorted(set(cat_arr.tolist())):
        mask = cat_arr == category
        if mask.sum() == 0:
            continue
        y_true_c = y_true[mask]
        y_prob_c = y_prob[mask]
        y_cls_c = y_cls[mask]
        acc_c = accuracy_score(y_true_c, y_cls_c)
        try:
            auc_c = roc_auc_score(y_true_c, y_prob_c) if len(np.unique(y_true_c)) > 1 else float('nan')
        except Exception:
            auc_c = float('nan')
        per_category[category] = {
            'count': int(mask.sum()),
            'accuracy': float(acc_c),
            'auc': float(auc_c),
        }
    metrics['per_category'] = per_category
    return metrics


def evaluate_end_to_end(model: EndToEndComparisonModel, dataloader: DataLoader, device: str) -> Dict[str, object]:
    """Evaluate end-to-end model"""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    steps = 0
    preds: List[float] = []
    labels: List[float] = []
    categories: List[str] = []
    left_ids: List[str] = []
    right_ids: List[str] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluate', leave=False):
            left_img = batch['left_image'].to(device)
            right_img = batch['right_image'].to(device)
            lab = batch['label'].float().to(device)
            logits = model(left_img, right_img, apply_sigmoid=False)
            prob = torch.sigmoid(logits)
            loss = criterion(logits, lab)
            total_loss += loss.item()
            steps += 1
            preds.extend(prob.cpu().numpy().tolist())
            labels.extend(lab.cpu().numpy().tolist())
            categories.extend(list(batch['category']))
            left_ids.extend(list(batch['left_id']))
            right_ids.extend(list(batch['right_id']))
    metrics = compute_pairwise_metrics(preds, labels, categories)
    metrics['loss'] = total_loss / max(1, steps)
    metrics['predictions'] = preds
    metrics['labels'] = labels
    metrics['categories'] = categories
    metrics['left_ids'] = left_ids
    metrics['right_ids'] = right_ids
    return metrics


SUMMARY_PATH = Path('result') / 'model_accuracy_summary.json'


def sanitize_float(value: float) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(round(value, 6))


def update_accuracy_summary(model_name: str, per_category: Dict[str, Dict[str, float]], summary_path: Path = SUMMARY_PATH) -> None:
    """Update each model's per-dimension accuracy to unified JSON for radar chart use"""
    summary: Dict[str, object]
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    else:
        summary = {'models': {}, 'metrics': []}
    
    metrics_existing = set(summary.get('metrics', []))
    metrics_new = set(per_category.keys())
    metrics_all = sorted(metrics_existing.union(metrics_new))
    summary['metrics'] = metrics_all
    
    model_entry = summary.setdefault('models', {}).get(model_name, {})
    model_entry = {k: sanitize_float(v) for k, v in model_entry.items()}
    
    for metric in metrics_all:
        if metric in per_category:
            model_entry[metric] = sanitize_float(per_category[metric].get('accuracy'))
        else:
            model_entry.setdefault(metric, None)
    
    summary['models'][model_name] = model_entry
    summary['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_predictions_csv(csv_path: Path, metrics: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['left_id', 'right_id', 'category', 'label', 'prob_left>right', 'pred_label'])
        for li, ri, cat, lab, prob in zip(
            metrics.get('left_ids', []),
            metrics.get('right_ids', []),
            metrics.get('categories', []),
            metrics.get('labels', []),
            metrics.get('predictions', []),
        ):
            writer.writerow([li, ri, cat, int(lab), float(prob), int(prob > 0.5)])




def create_grad_scaler() -> 'torch.cuda.amp.GradScaler':
    """Return AMP GradScaler compatible with different torch versions."""
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler(enabled=torch.cuda.is_available())
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


def autocast_context():
    """Return autocast context manager compatible with different torch versions."""
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available())
    return torch.cuda.amp.autocast(enabled=torch.cuda.is_available())
def save_metrics_json(json_path: Path, metrics: Dict[str, object], extra: Optional[Dict[str, object]] = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in metrics.items() if k not in ['predictions', 'labels', 'categories', 'left_ids', 'right_ids']}
    if extra:
        payload.update(extra)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_end_to_end_pipeline(args) -> None:
    """End-to-end baseline (CNN / ViT / CLIP) training entry point"""
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    config = EndToEndConfig(
        backbone=args.backbone,
        image_root=Path(args.image_root),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
        freeze_backbone=args.freeze_backbone,
        device=device,
        result_dir=Path(args.result_dir),
        log_dir=Path(args.log_dir) / 'end_to_end',
        max_samples=args.max_samples,
    )
    
    logger = create_logger(config.log_dir, f'{config.backbone}_baseline')
    logger.info(f'Starting end-to-end training: backbone={config.backbone}, device={device}')
    
    # 1) Read comparison data
    comparisons = extract_comparisons_from_graphs(args.graphs_dir, max_samples=config.max_samples)
    if not comparisons:
        logger.error('No comparison data found, terminating.')
        return
    
    # 2) Build image index & dataset
    try:
        image_index = build_image_index(config.image_root)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return
    
    try:
        feature_extractor, feature_dim, transform, backbone_name = build_feature_extractor(config.backbone, device)
    except Exception as exc:  # pragma: no cover - prompt when dependency is missing
        logger.error(f'Failed to build feature extractor: {exc}')
        return
    
    if config.freeze_backbone:
        logger.info('Feature backbone will remain frozen, only training Bradley-Terry head.')
    else:
        logger.info('Feature backbone will participate in end-to-end fine-tuning.')
    
    dataset = ImageComparisonDataset(comparisons, image_index, transform)
    if len(dataset) < 10:
        logger.error('Valid comparison count less than 10, cannot train.')
        return
    
    generator = torch.Generator().manual_seed(config.split_seed)
    train_len = max(1, int(len(dataset) * 0.7))
    val_len = max(1, int(len(dataset) * 0.15))
    test_len = len(dataset) - train_len - val_len
    if test_len <= 0:
        test_len = 1
        train_len = max(1, train_len - 1)
    splits = random_split(dataset, [train_len, val_len, test_len], generator=generator)
    train_set, val_set, test_set = splits
    
    workers = 0 if os.name == 'nt' else max(0, config.num_workers)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    full_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    
    # 3) Build model
    model = EndToEndComparisonModel(
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        hidden_dims=[512, 256, 128],
        dropout=0.1,
        freeze_backbone=config.freeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    logger.info(f'Training samples: {len(train_set)}, validation: {len(val_set)}, test: {len(test_set)}')
    logger.info(f'Feature extractor: {backbone_name}')
    
    best_state = None
    best_val_acc = -1.0
    best_val_metrics: Dict[str, object] = {}
    patience_counter = 0
    history: List[Dict[str, object]] = []
    
    scaler = create_grad_scaler()
    
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{config.num_epochs} - Train', leave=False):
            left_img = batch['left_image'].to(device)
            right_img = batch['right_image'].to(device)
            label = batch['label'].float().to(device)
            
            optimizer.zero_grad()
            with autocast_context():
                logits = model(left_img, right_img, apply_sigmoid=False)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            prob = torch.sigmoid(logits)
            pred_cls = (prob > 0.5).float()
            epoch_correct += (pred_cls == label).sum().item()
            epoch_total += label.size(0)
        
        train_loss = epoch_loss / max(1, len(train_loader))
        train_acc = epoch_correct / max(1, epoch_total)
        val_metrics = evaluate_end_to_end(model, val_loader, device)
        
        history.append({
            'epoch': epoch,
            'train_loss': sanitize_float(train_loss),
            'train_accuracy': sanitize_float(train_acc),
            'val_loss': sanitize_float(val_metrics.get('loss')),
            'val_accuracy': sanitize_float(val_metrics.get('accuracy')),
            'val_auc': sanitize_float(val_metrics.get('auc')),
        })
        logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_metrics["accuracy"]:.4f}, val_auc={val_metrics.get("auc", float("nan")):.4f}')
        
        current_val_acc = float(val_metrics.get('accuracy', 0.0))
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info('触发早停。')
                break
    
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_val_metrics = evaluate_end_to_end(model, val_loader, device)
    
    model.load_state_dict(best_state)
    model.to(device)
    logger.info('Loading model weights with best validation accuracy, starting test evaluation.')
    
    test_metrics = evaluate_end_to_end(model, test_loader, device)
    all_metrics = evaluate_end_to_end(model, full_loader, device)
    
    update_accuracy_summary(config.backbone, all_metrics.get('per_category', {}))
    
    result_dir = config.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    model_path = result_dir / f'{config.backbone}_baseline_best.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f'Saving model weights: {model_path}')
    
    save_predictions_csv(result_dir / f'predictions_{config.backbone}.csv', all_metrics)
    extra_metrics = {
        'split': {
            'train': len(train_set),
            'val': len(val_set),
            'test': len(test_set),
        },
        'val_accuracy': sanitize_float(best_val_metrics.get('accuracy')),
        'val_auc': sanitize_float(best_val_metrics.get('auc')),
        'test_accuracy': sanitize_float(test_metrics.get('accuracy')),
        'test_auc': sanitize_float(test_metrics.get('auc')),
        'backbone_name': backbone_name,
        'device': device,
    }
    save_metrics_json(result_dir / f'metrics_{config.backbone}.json', all_metrics, extra=extra_metrics)
    
    log_payload = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()},
        'best_val_accuracy': sanitize_float(best_val_acc),
        'test_accuracy': sanitize_float(test_metrics.get('accuracy')),
        'test_auc': sanitize_float(test_metrics.get('auc')),
        'history': history,
    }
    log_dir = config.log_dir / config.backbone
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'training_log.json', 'w', encoding='utf-8') as f:
        json.dump(log_payload, f, ensure_ascii=False, indent=2)
    logger.info('Training log saved.')

class BradleyTerryModel(nn.Module):
    """Bradley-Terry model - Learns latent score for each graph"""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2
    ):
        super(BradleyTerryModel, self).__init__()
        
        self.input_dim = input_dim
        
        # Score predictor
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.score_predictor = nn.Sequential(*layers)
    
    def forward(self, graph_repr):
        """Predict latent score of graph"""
        return self.score_predictor(graph_repr).squeeze()
    
    def predict_comparison(self, left_repr, right_repr):
        """Predict comparison result"""
        left_score = self.forward(left_repr)
        right_score = self.forward(right_repr)
        
        # Bradley-Terry model: P(left > right) = sigmoid(left_score - right_score)
        comparison_prob = torch.sigmoid(left_score - right_score)
        
        return comparison_prob


class ComparisonTrainer:
    """Comparison trainer"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            left_repr = batch['left_repr'].to(self.device)
            right_repr = batch['right_repr'].to(self.device)
            labels = batch['label'].float().to(self.device)
            
            # Forward pass
            predictions = self.model.predict_comparison(left_repr, right_repr)
            
            # Calculate loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted_labels = (predictions > 0.5).float()
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                left_repr = batch['left_repr'].to(self.device)
                right_repr = batch['right_repr'].to(self.device)
                labels = batch['label'].float().to(self.device)
                
                predictions = self.model.predict_comparison(left_repr, right_repr)
                
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                
                predicted_labels = (predictions > 0.5).float()
                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate AUC
        auc = roc_auc_score(all_labels, all_predictions)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
            'auc': auc,
            'predictions': all_predictions,
            'labels': all_labels
        }


def extract_comparisons_from_graphs(graphs_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Extract comparison information from graph data"""
    graphs_path = Path(graphs_dir)
    json_files = list(graphs_path.glob("graph_*.json"))
    
    if max_samples:
        json_files = json_files[:max_samples]
    
    comparisons = []
    
    for json_file in tqdm(json_files, desc="Extracting comparisons"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_id = data['file_id']
            comparisons_data = data.get('metadata', {}).get('comparisons', [])
            
            for comp in comparisons_data:
                comparison = {
                    'left_id': file_id if comp['role'] == 'left' else comp['opponent_id'],
                    'right_id': comp['opponent_id'] if comp['role'] == 'left' else file_id,
                    'winner': comp['winner'],
                    'category': comp['category'],
                    'left_lat': comp['lat'],
                    'left_long': comp['long'],
                    'right_lat': comp['opponent_lat'],
                    'right_long': comp['opponent_long']
                }
                comparisons.append(comparison)
                
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
            continue
    
    print(f"Extracted {len(comparisons)} comparisons")
    
    # Count category distribution
    category_counts = {}
    for comp in comparisons:
        category = comp['category']
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print("Comparison category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    return comparisons


def load_graph_representations(representations_file: str) -> Dict[str, torch.Tensor]:
    """Load graph representation vectors"""
    try:
        representations = torch.load(representations_file)
        print(f"Loaded {representations.shape[0]} graph representation vectors")
        
        # Need to map file_id to representation vectors
        # Here assumes representation vector order matches graph file order
        # In actual use, need to map based on file_id
        
        # Temporary solution: use index as ID
        repr_dict = {}
        for i, repr_vec in enumerate(representations):
            repr_dict[str(i)] = repr_vec
        
        return repr_dict
        
    except FileNotFoundError:
        print("Graph representation vector file not found, please run 05_graph_vae.py first")
        return {}


def plot_comparison_results(results: Dict[str, float], title: str = "Comparison Results"):
    """Plot comparison results"""
    categories = list(results.keys())
    values = list(results.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(categories, values)
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = Path('result') / 'category_auc.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_pr(labels: List[float], probs: List[float], out_dir: Path):
    y_true = np.array(labels).astype(int)
    y_score = np.array(probs)
    if len(set(y_true.tolist())) < 2:
        return
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC={auc:.3f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion(cm: List[List[int]], out_path: Path):
    if cm is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_with_breakdown(model: nn.Module, device: str, dataloader) -> Dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total = 0
    all_predictions: List[float] = []
    all_labels: List[int] = []
    all_categories: List[str] = []
    all_left_ids: List[str] = []
    all_right_ids: List[str] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate"):
            left_repr = batch['left_repr'].to(device)
            right_repr = batch['right_repr'].to(device)
            labels = batch['label'].float().to(device)
            predictions = model.predict_comparison(left_repr, right_repr)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            total += labels.size(0)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_categories.extend(list(batch['category']))
            all_left_ids.extend(list(batch['left_id']))
            all_right_ids.extend(list(batch['right_id']))
    # metrics
    try:
        auc = roc_auc_score(all_labels, all_predictions) if len(set(all_labels)) > 1 else float('nan')
    except Exception:
        auc = float('nan')
    try:
        ap = average_precision_score(all_labels, all_predictions) if len(set(all_labels)) > 1 else float('nan')
    except Exception:
        ap = float('nan')
    y_pred = (np.array(all_predictions) > 0.5).astype(int)
    acc = accuracy_score(all_labels, y_pred) if len(all_labels) else 0.0
    prec = precision_score(all_labels, y_pred, zero_division=0)
    rec = recall_score(all_labels, y_pred, zero_division=0)
    f1 = f1_score(all_labels, y_pred, zero_division=0)
    try:
        ll = log_loss(all_labels, np.clip(all_predictions, 1e-7, 1-1e-7))
    except Exception:
        ll = float('nan')
    try:
        brier = brier_score_loss(all_labels, all_predictions)
    except Exception:
        brier = float('nan')
    try:
        cm = confusion_matrix(all_labels, y_pred).tolist()
    except Exception:
        cm = None
    per_category: Dict[str, Dict[str, float]] = {}
    arr_pred = np.array(all_predictions)
    arr_lab = np.array(all_labels)
    arr_cat = np.array(all_categories)
    for c in sorted(set(all_categories)):
        idx = (arr_cat == c)
        if idx.sum() == 0:
            continue
        y_true_c = arr_lab[idx]
        y_prob_c = arr_pred[idx]
        y_cls_c = (y_prob_c > 0.5).astype(int)
        acc_c = accuracy_score(y_true_c, y_cls_c)
        try:
            auc_c = roc_auc_score(y_true_c, y_prob_c) if len(set(y_true_c.tolist())) > 1 else float('nan')
        except Exception:
            auc_c = float('nan')
        per_category[c] = {'count': int(idx.sum()), 'auc': float(auc_c), 'accuracy': float(acc_c)}
    return {
        'loss': total_loss / max(1, len(dataloader)),
        'accuracy': acc,
        'auc': auc,
        'ap': ap,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'logloss': ll,
        'brier': brier,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'categories': all_categories,
        'left_ids': all_left_ids,
        'right_ids': all_right_ids,
        'per_category': per_category,
    }


def load_graph_representations_with_ids(representations_file: str = 'graph_representations.pt', ids_file: str = 'graph_ids.json') -> Dict[str, torch.Tensor]:
    """Load graph representations + graph_ids.json, return file_id → representation dictionary."""
    try:
        reps = torch.load(representations_file)
        with open(ids_file, 'r', encoding='utf-8') as f:
            ids = json.load(f)
        if len(ids) != reps.shape[0]:
            print(f"Warning: Representation count ({reps.shape[0]}) and ID count ({len(ids)}) mismatch, aligning to minimum length.")
        n = min(len(ids), reps.shape[0])
        mapping: Dict[str, torch.Tensor] = {ids[i]: reps[i] for i in range(n)}
        print(f"Loaded {len(mapping)} (file_id → representation) mappings")
        return mapping
    except FileNotFoundError:
        print("Graph representation or ID mapping file not found. Please run 05_graph_vae.py first.")
        return {}
    except Exception as e:
        print(f"Failed to load graph representations: {e}")
        return {}


def filter_comparisons(comparisons: List[Dict], valid_ids: set) -> List[Dict]:
    """Keep only comparison samples where both IDs are in the representation mapping."""
    out = []
    drop = 0
    for c in comparisons:
        if c['left_id'] in valid_ids and c['right_id'] in valid_ids:
            out.append(c)
        else:
            drop += 1
    if drop:
        print(f"Filtered out {drop} samples missing representations, kept {len(out)} samples.")
    return out


def run_graphmae_pipeline(args) -> None:
    """GraphMAE representation + Bradley-Terry prediction head training pipeline"""
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    result_dir = Path(args.result_dir)
    log_dir = Path(args.log_dir) / 'graphmae'
    logger = create_logger(log_dir, 'graphmae_br')  # graph-based ranking
    logger.info(f'Starting GraphMAE → Bradley-Terry training, device: {device}')
    
    try:
        hidden_dims = [int(x.strip()) for x in args.graph_hidden_dims.split(',') if x.strip()]
        if not hidden_dims:
            raise ValueError
    except Exception:
        raise ValueError('--graph-hidden-dims parameter format error, e.g., 512,256,128')

    config = {
        'input_dim': args.graph_input_dim,
        'hidden_dims': hidden_dims,
        'learning_rate': args.graph_learning_rate,
        'weight_decay': args.graph_weight_decay,
        'early_stop_patience': args.graph_patience,
        'batch_size': args.graph_batch_size,
        'num_epochs': args.graph_epochs,
        'dropout': args.graph_dropout,
    }
    
    graph_representations = load_graph_representations_with_ids(args.repr_file, args.repr_ids)
    if not graph_representations:
        logger.error('Graph representation data not found, please run 05_graph_vae.py first.')
        return
    
    comparisons = extract_comparisons_from_graphs(args.graphs_dir, max_samples=args.max_samples)
    if not comparisons:
        logger.error('No comparison data found.')
        return
    comparisons = filter_comparisons(comparisons, set(graph_representations.keys()))
    if not comparisons:
        logger.error('No available comparison data after filtering.')
        return
    
    dataset = ComparisonDataset(comparisons, graph_representations)
    if len(dataset) < 10:
        logger.error('Valid comparison pairs less than 10, cannot train.')
        return
    
    generator = torch.Generator().manual_seed(args.split_seed)
    train_len = max(1, int(len(dataset) * 0.8))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)
    
    workers = 0 if os.name == 'nt' else max(0, args.num_workers)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=workers, pin_memory=pin_memory)
    full_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=workers, pin_memory=pin_memory)
    
    model = BradleyTerryModel(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        dropout=0.2,
    ).to(device)
    
    trainer = ComparisonTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=device
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode='max', factor=0.5, patience=5,
        min_lr=max(config['learning_rate'] / 20.0, 1e-6)
    )

    logger.info(f'Training set: {len(train_dataset)}, validation set: {len(val_dataset)}')
    
    best_auc = -1.0
    best_state = None
    best_val_metrics: Dict[str, float] = {}
    patience_counter = 0
    history: List[Dict[str, object]] = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        
        history.append({
            'epoch': epoch,
            'train_loss': sanitize_float(train_metrics.get('loss')),
            'train_accuracy': sanitize_float(train_metrics.get('accuracy')),
            'val_loss': sanitize_float(val_metrics.get('loss')),
            'val_accuracy': sanitize_float(val_metrics.get('accuracy')),
            'val_auc': sanitize_float(val_metrics.get('auc')),
        })
        
        logger.info(
            f'Epoch {epoch}: train_loss={train_metrics["loss"]:.4f}, '
            f'train_acc={train_metrics["accuracy"]:.4f}, '
            f'val_loss={val_metrics["loss"]:.4f}, '
            f'val_acc={val_metrics["accuracy"]:.4f}, '
            f'val_auc={val_metrics.get("auc", float("nan")):.4f}'
        )
        
        current_auc = float(val_metrics.get('auc', 0.0))
        scheduler.step(current_auc)

        if current_auc > best_auc:
            best_auc = current_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stop_patience']:
                logger.info('Early stopping triggered.')
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        logger.info('Loading model weights with best validation AUC.')
    else:
        logger.warning('Best model not updated, using weights from last epoch.')
    
    eval_val = evaluate_with_breakdown(model, device, val_loader)
    eval_all = evaluate_with_breakdown(model, device, full_loader)
    
    update_accuracy_summary('graphmae', eval_all.get('per_category', {}))
    
    result_dir.mkdir(parents=True, exist_ok=True)
    graph_model_path = result_dir / 'bradley_terry_graphmae_best.pth'
    torch.save(model.state_dict(), graph_model_path)
    logger.info(f'Saving model weights: {graph_model_path}')
    # Compatible with old pipeline: copy to root directory
    try:
        torch.save(model.state_dict(), Path('bradley_terry_comparison_best.pth'))
    except Exception as exc:  # pragma: no cover
        logger.warning(f'Failed to save root directory weights: {exc}')
    
    save_predictions_csv(result_dir / 'predictions_graphmae.csv', eval_all)
    extra_metrics = {
        'split': {'train': len(train_dataset), 'val': len(val_dataset)},
        'val_auc': sanitize_float(best_auc),
        'val_accuracy': sanitize_float(best_val_metrics.get('accuracy')),
        'device': device,
    }
    save_metrics_json(result_dir / 'metrics_graphmae.json', eval_all, extra=extra_metrics)
    
    log_payload = {
        'config': {k: v for k, v in config.items()},
        'history': history,
        'best_val_auc': sanitize_float(best_auc),
        'best_val_accuracy': sanitize_float(best_val_metrics.get('accuracy')),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'training_log.json', 'w', encoding='utf-8') as f:
        json.dump(log_payload, f, ensure_ascii=False, indent=2)
    logger.info('GraphMAE + Bradley-Terry training complete, log saved.')


def parse_args():
    parser = argparse.ArgumentParser(description='Comparison-based perception prediction training script')
    parser.add_argument('--backbone', choices=['graphmae', 'cnn', 'vit', 'clip'], default='graphmae',
                        help='Feature extraction backbone model to use')
    parser.add_argument('--graphs-dir', type=str, default='output/stage_03_scene_graphs', help='Scene graph JSON directory')
    parser.add_argument('--repr-file', type=str, default='graph_representations.pt', help='GraphMAE representation file path')
    parser.add_argument('--repr-ids', type=str, default='graph_ids.json', help='GraphMAE representation corresponding ID JSON')
    parser.add_argument('--image-root', type=str, default='data/sample', help='Image root directory (for end-to-end baseline)')
    parser.add_argument('--result-dir', type=str, default='result', help='Result output directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Training log directory')
    parser.add_argument('--batch-size', type=int, default=32, help='End-to-end baseline training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='End-to-end baseline training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='End-to-end baseline learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-5, help='End-to-end baseline weight decay')
    parser.add_argument('--patience', type=int, default=20, help='End-to-end baseline early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--split-seed', type=int, default=42, help='Data split random seed')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of comparison samples to read')
    parser.add_argument('--device', type=str, default=None, help='Force specify training device (cpu / cuda)')
    parser.add_argument('--freeze-backbone', dest='freeze_backbone', action='store_true', default=False, help='Freeze end-to-end baseline feature backbone')
    parser.add_argument('--train-backbone', dest='freeze_backbone', action='store_false', help='Allow end-to-end baseline to fine-tune feature backbone (default enabled)')
    # GraphMAE specific configuration
    parser.add_argument('--graph-batch-size', type=int, default=32, help='GraphMAE training batch size')
    parser.add_argument('--graph-epochs', type=int, default=220, help='GraphMAE training epochs')
    parser.add_argument('--graph-learning-rate', type=float, default=1e-4, help='GraphMAE learning rate')
    parser.add_argument('--graph-weight-decay', type=float, default=5e-5, help='GraphMAE weight decay')
    parser.add_argument('--graph-patience', type=int, default=40, help='GraphMAE early stopping patience')
    parser.add_argument('--graph-input-dim', type=int, default=128, help='GraphMAE representation dimension')
    parser.add_argument('--graph-hidden-dims', type=str, default='512,256,128', help='Bradley-Terry head hidden layer configuration, comma-separated')
    parser.add_argument('--graph-dropout', type=float, default=0.1, help='Bradley-Terry head dropout')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.backbone == 'graphmae':
        run_graphmae_pipeline(args)
    else:
        run_end_to_end_pipeline(args)


if __name__ == "__main__":
    main()
