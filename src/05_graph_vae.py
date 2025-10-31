#!/usr/bin/env python3
"""
GraphMAE-style pretraining for scene graphs with masked node/edge feature
reconstruction and graph-level summary alignment.

- Encoder: GINEConv stacks with edge attributes
- Node task: reconstruct masked node features (MSE on masked positions)
- Edge task: reconstruct masked edge features (MSE on masked edges)
- Graph task: predict summary embedding (`graph_attr` / `summary_embedding`)

Outputs a graph-level embedding suitable for downstream comparison training.
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool
from tqdm import tqdm


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    ]
    return nn.Sequential(*layers)


class GraphMAE(nn.Module):
    """Masked Graph Autoencoder for feature reconstruction and graph alignment - GPU accelerated version"""

    def __init__(
        self,
        node_feature_dim: int = 384,
        edge_feature_dim: int = 384,
        graph_feature_dim: int = 384,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.graph_feature_dim = graph_feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.device = self._get_device(device)

        # Input projections
        self.node_in = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_in = nn.Linear(edge_feature_dim, hidden_dim)

        # Learnable mask tokens
        self.node_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.edge_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.node_mask_token, std=0.02)
        nn.init.normal_(self.edge_mask_token, std=0.02)

        # GINE layers (use edge_attr via edge_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            gin_mlp = build_mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
            conv = GINEConv(gin_mlp, train_eps=True, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity())

        # Decoders
        self.node_decoder = build_mlp(hidden_dim, hidden_dim, node_feature_dim, dropout)
        # For edge reconstruction, use [h_src || h_dst || edge_ctx]
        self.edge_decoder = build_mlp(hidden_dim * 3, hidden_dim, edge_feature_dim, dropout)

        # Graph-level heads
        self.graph_projector = build_mlp(hidden_dim * 2, hidden_dim, latent_dim, dropout)
        self.graph_pred_head = build_mlp(hidden_dim * 2, hidden_dim, graph_feature_dim, dropout)
        
        # Move to specified device after initialization
        self.to(self.device)
        if self.device == "cuda":
            self._setup_gpu_memory()
    
    def _get_device(self, device: Optional[str]) -> str:
        """Get computing device"""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                device = "cuda"
                print(f"✓ CUDA detected, using GPU acceleration (Device: {torch.cuda.get_device_name()})")
            else:
                device = "cpu"
                print("⚠ CUDA not available, using CPU computation")
        elif device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        return device
    
    def _setup_gpu_memory(self):
        """Setup GPU memory management"""
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Get GPU memory information
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                cached_memory = torch.cuda.memory_reserved(0)
                
                print(f"GPU memory information:")
                print(f"  Total memory: {total_memory / 1024**3:.2f} GB")
                print(f"  Allocated: {allocated_memory / 1024**3:.2f} GB")
                print(f"  Cached: {cached_memory / 1024**3:.2f} GB")
                
        except Exception as e:
            print(f"GPU memory setup warning: {e}")

    def encode_nodes(self, x_h: torch.Tensor, edge_index: torch.Tensor, e_h: torch.Tensor) -> torch.Tensor:
        h = x_h
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index, e_h)
            out = norm(out)
            out = F.relu(out)
            h = h + out  # residual
        return h

    def graph_readout(self, h: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        mean_pooled = global_mean_pool(h, batch_index)
        max_pooled = global_max_pool(h, batch_index)
        return torch.cat([mean_pooled, max_pooled], dim=-1)

    def forward(
        self,
        batch: Data,
        node_mask_ratio: float = 0.3,
        edge_mask_ratio: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        # Project inputs
        x_h = self.node_in(batch.x)
        if batch.edge_attr is not None and batch.edge_attr.numel() > 0:
            e_h = self.edge_in(batch.edge_attr)
        else:
            e_h = torch.zeros((batch.edge_index.size(1), self.hidden_dim), device=batch.x.device, dtype=x_h.dtype)

        # Build masks per-graph
        node_mask = self._sample_node_mask(batch.batch, node_mask_ratio)
        if batch.edge_index.size(1) > 0 and edge_mask_ratio > 0:
            edge_mask = self._sample_edge_mask(batch, edge_mask_ratio)
        else:
            edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.x.device)

        # Apply masking tokens
        x_h_masked = x_h.clone()
        x_h_masked[node_mask] = self.node_mask_token
        e_h_masked = e_h.clone()
        if edge_mask.any():
            e_h_masked[edge_mask] = self.edge_mask_token

        # Encode nodes with masked inputs
        h_nodes = self.encode_nodes(x_h_masked, batch.edge_index, e_h_masked)

        # Graph embedding and prediction
        g_context = self.graph_readout(h_nodes, batch.batch)
        z_graph = self.graph_projector(g_context)
        graph_pred = self.graph_pred_head(g_context)
        graph_pred_norm = F.normalize(graph_pred, dim=-1, eps=1e-8)

        # Node reconstruction on masked nodes
        node_recon_loss = torch.tensor(0.0, device=batch.x.device)
        num_masked_nodes = int(node_mask.sum().item())
        if num_masked_nodes > 0:
            node_pred = self.node_decoder(h_nodes[node_mask])
            node_recon_loss = F.mse_loss(node_pred, batch.x[node_mask])

        # Edge reconstruction on masked edges
        edge_recon_loss = torch.tensor(0.0, device=batch.x.device)
        if int(edge_mask.sum().item()) > 0:
            masked_idx = torch.nonzero(edge_mask, as_tuple=False).view(-1)
            src = batch.edge_index[0, masked_idx]
            dst = batch.edge_index[1, masked_idx]
            h_src = h_nodes[src]
            h_dst = h_nodes[dst]
            e_ctx = e_h_masked[masked_idx]
            dec_in = torch.cat([h_src, h_dst, e_ctx], dim=-1)
            edge_pred = self.edge_decoder(dec_in)
            edge_recon_loss = F.mse_loss(edge_pred, batch.edge_attr[masked_idx])

        # Graph summary alignment
        graph_align_loss = torch.tensor(0.0, device=batch.x.device)
        graph_attr_norm = None
        graph_attr_raw = None
        if hasattr(batch, 'graph_attr') and batch.graph_attr is not None:
            graph_attr = batch.graph_attr.to(graph_pred.device)
            if graph_attr.dim() == 1:
                graph_attr = graph_attr.view(1, -1)
            graph_attr = graph_attr.view(-1, graph_pred.size(-1))
            graph_attr_raw = graph_attr
            pred_n = F.normalize(graph_pred, dim=-1, eps=1e-8)
            target_n = F.normalize(graph_attr, dim=-1, eps=1e-8)
            cos_loss = 1 - (pred_n * target_n).sum(dim=-1).mean()
            mse_loss = F.mse_loss(graph_pred, graph_attr)
            graph_align_loss = 0.8 * cos_loss + 0.2 * mse_loss
            graph_attr_norm = target_n

        return {
            'z_graph': z_graph,
            'node_mask': node_mask,
            'edge_mask': edge_mask,
            'node_recon_loss': node_recon_loss,
            'edge_recon_loss': edge_recon_loss,
            'graph_align_loss': graph_align_loss,
            'graph_pred': graph_pred,
            'graph_pred_norm': graph_pred_norm,
            'graph_attr_norm': graph_attr_norm,
            'graph_attr_raw': graph_attr_raw,
        }

    @torch.no_grad()
    def encode_graph(self, batch: Data) -> torch.Tensor:
        # No masking, clean inputs
        x_h = self.node_in(batch.x)
        if batch.edge_attr is not None and batch.edge_attr.numel() > 0:
            e_h = self.edge_in(batch.edge_attr)
        else:
            e_h = torch.zeros((batch.edge_index.size(1), self.hidden_dim), device=batch.x.device, dtype=x_h.dtype)
        h_nodes = self.encode_nodes(x_h, batch.edge_index, e_h)
        g_context = self.graph_readout(h_nodes, batch.batch)
        return self.graph_projector(g_context)

    def _sample_node_mask(self, batch_index: torch.Tensor, ratio: float) -> torch.Tensor:
        device = batch_index.device
        num_nodes = batch_index.size(0)
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if ratio <= 0:
            return mask
        num_graphs = int(batch_index.max().item()) + 1
        for gid in range(num_graphs):
            idx = torch.nonzero(batch_index == gid, as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            k = max(1, math.ceil(ratio * idx.numel()))
            choice = idx[torch.randperm(idx.numel(), device=device)[:k]]
            mask[choice] = True
        return mask

    def _sample_edge_mask(self, batch: Data, ratio: float) -> torch.Tensor:
        device = batch.edge_index.device
        num_edges = batch.edge_index.size(1)
        mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        if ratio <= 0 or num_edges == 0:
            return mask
        # Group edges by graph via source node's batch id
        src_graph = batch.batch[batch.edge_index[0]]
        num_graphs = int(src_graph.max().item()) + 1
        for gid in range(num_graphs):
            eidx = torch.nonzero(src_graph == gid, as_tuple=False).view(-1)
            if eidx.numel() == 0:
                continue
            k = max(1, math.ceil(ratio * eidx.numel()))
            choice = eidx[torch.randperm(eidx.numel(), device=device)[:k]]
            mask[choice] = True
        return mask


class GraphMAETrainer:
    """Trainer for GraphMAE pretraining - GPU accelerated version"""

    def __init__(
        self,
        model: GraphMAE,
        learning_rate: float = 1e-3,
        lambda_edge: float = 1.0,
        lambda_graph: float = 0.3,
        lambda_contrastive: float = 1.0,
        contrastive_temperature: float = 0.07,
        node_mask_ratio: float = 0.3,
        edge_mask_ratio: float = 0.3,
        device: Optional[str] = None
    ):
        # Use model's device, auto-detect if not specified
        if device is None:
            device = model.device
        else:
            device = self._get_device(device)
        
        self.model = model.to(device)
        self.device = device
        self.lambda_edge = lambda_edge
        self.lambda_graph = lambda_graph
        self.lambda_contrastive = lambda_contrastive
        self.contrastive_temperature = contrastive_temperature
        self.node_mask_ratio = node_mask_ratio
        self.edge_mask_ratio = edge_mask_ratio
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # GPU memory management
        if self.device == "cuda":
            self._setup_gpu_memory()
    
    def _get_device(self, device: Optional[str]) -> str:
        """Get computing device"""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                device = "cuda"
                print(f"✓ CUDA detected, using GPU acceleration (Device: {torch.cuda.get_device_name()})")
            else:
                device = "cpu"
                print("⚠ CUDA not available, using CPU computation")
        elif device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        return device
    
    def _setup_gpu_memory(self):
        """Setup GPU memory management"""
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Get GPU memory information
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                cached_memory = torch.cuda.memory_reserved(0)
                
                print(f"GPU memory information:")
                print(f"  Total memory: {total_memory / 1024**3:.2f} GB")
                print(f"  Allocated: {allocated_memory / 1024**3:.2f} GB")
                print(f"  Cached: {cached_memory / 1024**3:.2f} GB")
                
        except Exception as e:
            print(f"GPU memory setup warning: {e}")

    def compute_loss(self, out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        node_loss = out['node_recon_loss']
        edge_loss = out['edge_recon_loss']
        graph_loss = out['graph_align_loss']
        contrastive_loss = torch.tensor(0.0, device=node_loss.device)
        if self.lambda_contrastive > 0:
            graph_attr_norm = out.get('graph_attr_norm')
            graph_pred_norm = out.get('graph_pred_norm')
            if (
                graph_attr_norm is not None
                and graph_pred_norm is not None
                and graph_attr_norm.size(0) > 1
            ):
                logits = graph_pred_norm @ graph_attr_norm.t()
                logits = logits / self.contrastive_temperature
                targets = torch.arange(logits.size(0), device=logits.device)
                loss_i = F.cross_entropy(logits, targets)
                loss_t = F.cross_entropy(logits.t(), targets)
                contrastive_loss = 0.5 * (loss_i + loss_t)
        total = (
            node_loss
            + self.lambda_edge * edge_loss
            + self.lambda_graph * graph_loss
            + self.lambda_contrastive * contrastive_loss
        )
        metrics = {
            'total': float(total.detach().cpu().item()),
            'node': float(node_loss.detach().cpu().item()),
            'edge': float(edge_loss.detach().cpu().item()),
            'graph': float(graph_loss.detach().cpu().item()),
            'contrastive': float(contrastive_loss.detach().cpu().item()),
        }
        return total, metrics

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        running = {'total': 0.0, 'node': 0.0, 'edge': 0.0, 'graph': 0.0, 'contrastive': 0.0}
        for batch in tqdm(dataloader, desc="Training"):
            batch = batch.to(self.device)
            out = self.model(batch, self.node_mask_ratio, self.edge_mask_ratio)
            loss, metrics = self.compute_loss(out)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for k in running:
                running[k] += metrics[k]
            
            # GPU memory management - periodic cache clearing
            if self.device == "cuda" and hasattr(self, '_step_count'):
                self._step_count += 1
                if self._step_count % 10 == 0:  # Clear every 10 steps
                    torch.cuda.empty_cache()
            elif self.device == "cuda":
                self._step_count = 1
        
        n = max(1, len(dataloader))
        return {k: v / n for k, v in running.items()}

    @torch.no_grad()
    def validate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        running = {'total': 0.0, 'node': 0.0, 'edge': 0.0, 'graph': 0.0, 'contrastive': 0.0}
        for batch in tqdm(dataloader, desc="Validation"):
            batch = batch.to(self.device)
            out = self.model(batch, self.node_mask_ratio, self.edge_mask_ratio)
            loss, metrics = self.compute_loss(out)
            for k in running:
                running[k] += metrics[k]
        n = max(1, len(dataloader))
        return {k: v / n for k, v in running.items()}

    @torch.no_grad()
    def extract_representations(self, dataloader) -> Tuple[torch.Tensor, List[str]]:
        self.model.eval()
        reps = []
        ids: List[str] = []
        for batch in tqdm(dataloader, desc="Extracting representations"):
            batch = batch.to(self.device)
            z = self.model.encode_graph(batch)
            reps.append(z.cpu())
            # file_id collates to list when batching
            if hasattr(batch, 'file_id') and batch.file_id is not None:
                file_ids = batch.file_id
                if isinstance(file_ids, (list, tuple)):
                    ids.extend(file_ids)
                else:
                    ids.extend([file_ids] * batch.num_graphs)
            else:
                ids.extend([None] * batch.num_graphs)
        return torch.cat(reps, dim=0), ids


def sanitize_graph_attributes(graphs: List[Data]) -> None:
    """Remove optional text-heavy fields to keep collate keys consistent."""
    optional_attrs = ['entity_texts', 'relation_texts', 'summary', 'summary_embedding']
    for graph in graphs:
        for attr in optional_attrs:
            if hasattr(graph, attr):
                try:
                    delattr(graph, attr)
                except AttributeError:
                    graph.__dict__.pop(attr, None)


def pack_graphs_to_dataset(data_dir: str, output_file: str, max_samples: Optional[int] = None) -> bool:
    """Pack multiple graph files into a single dataset file"""
    data_path = Path(data_dir)
    output_path = Path(output_file)
    
    # Check if packed file already exists
    if output_path.exists():
        print(f"✓ Found existing packed file: {output_path}")
        return True
    
    print(f"Starting to pack graph data to: {output_path}")
    
    # Find all graph files
    pt_files = sorted(data_path.glob("graph_*.pt"))
    if not pt_files:
        pt_files = sorted(data_path.glob("*.pt"))
    
    if not pt_files:
        print(f"No .pt files found in {data_path}")
        return False
    
    if max_samples and pt_files:
        pt_files = pt_files[:max_samples]
    
    print(f"Found {len(pt_files)} graph files")
    
    graphs: List[Data] = []
    failed_count = 0
    
    for pt_file in tqdm(pt_files, desc="Packing graph data"):
        try:
            try:
                graph = torch.load(pt_file, map_location="cpu", weights_only=False)
            except TypeError:
                graph = torch.load(pt_file, map_location="cpu")
            
            if isinstance(graph, dict):
                graph = Data(**graph)
            
            if not isinstance(graph, Data):
                print(f"Skipping {pt_file}: not a valid PyG Data object")
                failed_count += 1
                continue
            
            if graph.x is None or graph.x.numel() == 0:
                print(f"Skipping {pt_file}: empty graph")
                failed_count += 1
                continue
            
            # Ensure file_id exists
            if getattr(graph, "file_id", None) is None:
                stem = pt_file.stem
                file_id = stem[len("graph_"):] if stem.startswith("graph_") else stem
                graph.file_id = file_id
            
            graphs.append(graph.to("cpu"))
            
        except Exception as e:
            print(f"Error reading file {pt_file}: {e}")
            failed_count += 1
            continue
    
    if not graphs:
        print("No graph data successfully loaded")
        return False
    
    sanitize_graph_attributes(graphs)
    
    # Save packed dataset
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graphs, str(output_path))
        print(f"✓ Successfully packed {len(graphs)} graphs to {output_path}")
        if failed_count > 0:
            print(f"⚠ Skipped {failed_count} invalid files")
        return True
    except Exception as e:
        print(f"Failed to save packed file: {e}")
        return False


def load_graph_data(data_dir: str, max_samples: Optional[int] = None, force_pack: bool = False) -> List[Data]:
    """Load PyG graph tensors (.pt); fallback to JSON parsing if needed."""
    data_path = Path(data_dir)
    
    # First try to load packed dataset file
    packed_file = data_path / "graphs_dataset.pt"
    if packed_file.exists() and not force_pack:
        print(f"✓ Loading packed dataset: {packed_file}")
        try:
            graphs = torch.load(packed_file, map_location="cpu", weights_only=False)
            if isinstance(graphs, list) and len(graphs) > 0:
                sanitize_graph_attributes(graphs)
                if max_samples:
                    graphs = graphs[:max_samples]
                print(f"✓ Loaded {len(graphs)} graphs from packed file")
                return graphs
        except Exception as e:
            print(f"Failed to load packed file: {e}")
    
    # If no packed file or force repack, try packing existing files
    if force_pack and packed_file.exists():
        print("Force repacking, deleting existing packed file...")
        packed_file.unlink()
    
    if pack_graphs_to_dataset(data_dir, str(packed_file), max_samples):
        try:
            graphs = torch.load(packed_file, map_location="cpu", weights_only=False)
            if isinstance(graphs, list) and len(graphs) > 0:
                sanitize_graph_attributes(graphs)
                print(f"✓ Loaded {len(graphs)} graphs from newly packed file")
                return graphs
        except Exception as e:
            print(f"Failed to load newly packed file: {e}")
    
    # If packing fails, fall back to loading one by one
    print("Falling back to loading graph files one by one...")
    pt_files = sorted(data_path.glob("graph_*.pt"))
    if not pt_files:
        pt_files = sorted(data_path.glob("*.pt"))
    if max_samples and pt_files:
        pt_files = pt_files[:max_samples]

    graphs: List[Data] = []

    if pt_files:
        for pt_file in tqdm(pt_files, desc="Loading graphs"):
            try:
                try:
                    graph = torch.load(pt_file, map_location="cpu", weights_only=False)
                except TypeError:
                    graph = torch.load(pt_file, map_location="cpu")
                if isinstance(graph, dict):
                    graph = Data(**graph)
                if not isinstance(graph, Data):
                    print(f"Skipping {pt_file}: not a valid PyG Data object")
                    continue
                if graph.x is None or graph.x.numel() == 0:
                    continue
                if getattr(graph, "file_id", None) is None:
                    stem = pt_file.stem
                    file_id = stem[len("graph_"):] if stem.startswith("graph_") else stem
                    graph.file_id = file_id
                graphs.append(graph.to("cpu"))
            except Exception as e:
                print(f"Error reading file {pt_file}: {e}")
                continue
        return graphs

    # Fallback to legacy JSON loading when .pt files are unavailable.
    json_files = sorted(data_path.glob("graph_*.json"))
    if not json_files:
        json_files = sorted(data_path.glob("*.json"))
    mismatch_count = 0
    empty_graph_count = 0
    if max_samples:
        json_files = json_files[:max_samples]

    for json_file in tqdm(json_files, desc="Loading graphs"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            required_keys = {'node_features', 'edge_index', 'edge_features', 'graph_features', 'file_id'}
            if not required_keys.issubset(data.keys()):
                print(f"Skipping {json_file}: missing required fields {required_keys - set(data.keys())}")
                continue

            node_features = torch.tensor(data['node_features'], dtype=torch.float32)
            if node_features.numel() == 0:
                empty_graph_count += 1
                continue
            edge_index_raw = data['edge_index']
            edge_features_raw = data['edge_features']
            edge_feature_dim = len(edge_features_raw[0]) if edge_features_raw else len(data['graph_features'])
            edge_index = torch.tensor(edge_index_raw, dtype=torch.long)
            edge_features = torch.tensor(edge_features_raw, dtype=torch.float32)
            if edge_features.dim() == 1 and edge_features.numel() > 0:
                edge_features = edge_features.view(1, -1)
            if edge_features.numel() == 0:
                edge_features = torch.zeros((0, edge_feature_dim), dtype=torch.float32)
            graph_features = torch.tensor(data['graph_features'], dtype=torch.float32)
            if edge_index.dim() == 1:
                edge_index = edge_index.view(2, -1)
            if edge_index.size(0) != 2:
                edge_index = edge_index.t() if edge_index.size(-1) == 2 else edge_index
            num_edges_idx = edge_index.size(1)
            num_edges_attr = edge_features.size(0)
            if num_edges_attr != num_edges_idx:
                mismatch_count += 1
                min_edges = min(num_edges_attr, num_edges_idx)
                if min_edges == 0:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    edge_features = torch.zeros((0, edge_features.size(-1) if edge_features.numel() > 0 else 0), dtype=edge_features.dtype)
                else:
                    edge_index = edge_index[:, :min_edges]
                    edge_features = edge_features[:min_edges]

            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features,
                graph_attr=graph_features,
                file_id=data['file_id']
            )

            graphs.append(graph)

        except Exception as e:
            print(f"Error reading file {json_file}: {e}")
            continue

    if mismatch_count:
        print(f"Automatically corrected edge count mismatch in {mismatch_count} graphs.")
    if empty_graph_count:
        print(f"Skipped {empty_graph_count} graph data with no nodes.")

    return graphs


def create_dataloader(graphs: List[Data], batch_size: int = 32, shuffle: bool = True, device: str = "cpu"):
    """Create PyG DataLoader for list of graphs with GPU optimization."""
    from torch_geometric.loader import DataLoader
    
    # Use larger batch size on GPU
    if device == "cuda":
        batch_size = min(batch_size * 2, 64)  # GPU can handle larger batches
        print(f"✓ Using GPU batch processing, batch size: {batch_size}")
    else:
        print(f"✓ Using CPU batch processing, batch size: {batch_size}")
    
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


def main():
    """Run GraphMAE pretraining and export graph embeddings - GPU accelerated version."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GraphMAE pretraining - GPU acceleration support")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cuda", "cpu"],
                       help="Computing device (auto: auto-detect, cuda: GPU, cpu: CPU)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples (None means use all)")
    parser.add_argument("--lambda-contrastive", type=float, default=1.0,
                       help="Contrastive loss weight")
    parser.add_argument("--contrastive-temperature", type=float, default=0.07,
                       help="Contrastive loss temperature coefficient")
    parser.add_argument("--data-dir", type=str, default="output/stage_04_pytorch",
                       help="Graph data directory")
    parser.add_argument("--output-dir", type=str, default="./packed",
                       help="Output directory")
    parser.add_argument("--force-pack", action="store_true",
                       help="Force repack data (even if packed file already exists)")
    
    args = parser.parse_args()
    
    config = {
        'node_feature_dim': 384,
        'edge_feature_dim': 384,
        'graph_feature_dim': 384,
        'hidden_dim': 256,
        'latent_dim': 128,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.0003,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'max_samples': args.max_samples,
        'node_mask_ratio': 0.3,
        'edge_mask_ratio': 0.3,
        'lambda_edge': 1.0,
        'lambda_graph': 0.3,
        'lambda_contrastive': args.lambda_contrastive,
        'contrastive_temperature': args.contrastive_temperature,
        'seed': 26,
        'early_stop_patience': 20,
        'device': args.device,
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'force_pack': args.force_pack,
    }

    print("=== GraphMAE Pretraining (GPU Accelerated) ===")
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Training epochs: {config['num_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Loss weights: λ_edge={config['lambda_edge']}, λ_graph={config['lambda_graph']}, λ_contrastive={config['lambda_contrastive']}, τ={config['contrastive_temperature']}")
    print(f"Data directory: {config['data_dir']}")
    print(f"Output directory: {config['output_dir']}")
    print()

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    print("Loading graph data...")
    graphs = load_graph_data(config['data_dir'], max_samples=config['max_samples'], force_pack=config['force_pack'])
    print(f"Loaded {len(graphs)} graphs in total")

    if len(graphs) == 0:
        print("No graph data found, exiting.")
        return

    random.shuffle(graphs)

    file_ids = [getattr(graph, 'file_id', None) for graph in graphs]
    unique_file_ids = {fid for fid in file_ids if fid is not None}
    if len(unique_file_ids) != len([fid for fid in file_ids if fid is not None]):
        print("Warning: Duplicate file_id found in dataset, need to confirm if it causes data leakage.")

    # Split
    train_size = int(0.8 * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]

    train_loader = create_dataloader(train_graphs, batch_size=config['batch_size'], shuffle=True, device=config['device'])
    val_loader = create_dataloader(val_graphs, batch_size=config['batch_size'], shuffle=False, device=config['device'])

    train_ids = {getattr(graph, 'file_id', None) for graph in train_graphs}
    val_ids = {getattr(graph, 'file_id', None) for graph in val_graphs}
    leakage_ids = (train_ids & val_ids) - {None}
    if leakage_ids:
        print(f"Warning: Training and validation sets have {len(leakage_ids)} duplicate file_ids, possible data leakage.")
    else:
        print("Data split check: training/validation set file_ids have no overlap.")

    # Model
    model = GraphMAE(
        node_feature_dim=config['node_feature_dim'],
        edge_feature_dim=config['edge_feature_dim'],
        graph_feature_dim=config['graph_feature_dim'],
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        device=config['device'],
    )

    trainer = GraphMAETrainer(
        model=model,
        learning_rate=config['learning_rate'],
        lambda_edge=config['lambda_edge'],
        lambda_graph=config['lambda_graph'],
        lambda_contrastive=config['lambda_contrastive'],
        contrastive_temperature=config['contrastive_temperature'],
        node_mask_ratio=config['node_mask_ratio'],
        edge_mask_ratio=config['edge_mask_ratio'],
        device=config['device'],
    )

    print("Starting GraphMAE pretraining...")
    best_val = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    patience = config.get('early_stop_patience')
    for epoch in range(config['num_epochs']):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        print(
            f"Epoch {epoch+1}/{config['num_epochs']}\n"
            f"Train: total={train_metrics['total']:.4f}, node={train_metrics['node']:.4f}, edge={train_metrics['edge']:.4f}, graph={train_metrics['graph']:.4f}, contrastive={train_metrics['contrastive']:.4f}\n"
            f"Val:   total={val_metrics['total']:.4f}, node={val_metrics['node']:.4f}, edge={val_metrics['edge']:.4f}, graph={val_metrics['graph']:.4f}, contrastive={val_metrics['contrastive']:.4f}"
        )
        if val_metrics['total'] < best_val:
            best_val = val_metrics['total']
            best_epoch = epoch + 1
            epochs_no_improve = 0
            model_path = output_dir / 'graph_mae_best.pth'
            torch.save(model.state_dict(), str(model_path))
            print(f"Saved best model: {model_path}")
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= patience:
                print(f"Early stopping triggered: {patience} consecutive epochs without improvement (best epoch = {best_epoch}, best val loss = {best_val:.4f}).")
                break
        
        # GPU memory management - clear after each epoch
        if config['device'] == "cuda":
            torch.cuda.empty_cache()
        
        print("-" * 50)

    # Extract representations on all graphs
    print("Extracting graph-level representations...")
    full_loader = create_dataloader(graphs, batch_size=config['batch_size'], shuffle=False, device=config['device'])
    reps, ids = trainer.extract_representations(full_loader)
    print(f"Obtained {reps.shape[0]} graph representations, dimension {reps.shape[1]}")

    # Save tensor and id mapping (tensor for existing trainer; ids for proper mapping)
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reps_path = output_dir / 'graph_representations.pt'
    ids_path = output_dir / 'graph_ids.json'
    
    torch.save(reps, str(reps_path))
    try:
        with open(str(ids_path), 'w', encoding='utf-8') as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)
        print(f"Saved graph ID mapping: {ids_path}")
    except Exception as e:
        print(f"Failed to save graph ID mapping: {e}")
    print(f"Graph representations saved to {reps_path}")
    
    # Final GPU memory cleanup
    if config['device'] == "cuda":
        torch.cuda.empty_cache()
        print("✓ GPU memory cleared")


if __name__ == "__main__":
    main()
