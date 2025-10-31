#!/usr/bin/env python3
"""
Convert Sentence-BERT scene graph JSON files to PyTorch Geometric .pt files
"""

import argparse
import concurrent.futures
import json
import os
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Try importing PyTorch Geometric
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: Unable to import torch_geometric")


class PyTorchGraphBuilder:
    """PyTorch Geometric graph builder"""
    
    def __init__(self):
        """Initialize graph builder"""
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric library is required")
    
    @staticmethod
    def load_json_graph(json_file: Path) -> Optional[Dict]:
        """Load JSON graph file"""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Failed to load file {json_file}: {e}")
            return None
    
    @staticmethod
    def json_to_pytorch_data(json_data: Dict) -> Optional[Data]:
        """Convert JSON data to PyTorch Geometric Data object"""
        try:
            required_keys = {"node_features", "edge_index", "edge_features", "graph_features"}
            if not required_keys.issubset(json_data.keys()):
                missing = ', '.join(sorted(required_keys.difference(json_data.keys())))
                print(f"Skipping: JSON missing required fields -> {missing}")
                return None
            
            # Extract features
            node_features = torch.tensor(json_data["node_features"], dtype=torch.float32)
            edge_index = torch.tensor(json_data["edge_index"], dtype=torch.long)
            edge_features = torch.tensor(json_data["edge_features"], dtype=torch.float32)
            graph_features = torch.tensor(json_data["graph_features"], dtype=torch.float32)
            
            # Ensure edge_index has correct shape
            if edge_index.shape[0] != 2:
                edge_index = edge_index.T
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=node_features,                    # Node features [num_nodes, feature_dim]
                edge_index=edge_index,              # Edge index [2, num_edges]
                edge_attr=edge_features,            # Edge features [num_edges, feature_dim]
                graph_attr=graph_features,          # Graph-level features [feature_dim]
                num_nodes=json_data["num_nodes"],   # Number of nodes
                num_edges=json_data["num_edges"]    # Number of edges
            )
            
            # Add extra information
            data.summary = json_data.get("summary", "")
            summary_embedding = json_data.get("summary_embedding", [])
            data.summary_embedding = torch.tensor(summary_embedding, dtype=torch.float32) if summary_embedding else torch.empty(0, dtype=torch.float32)
            data.file_id = json_data.get("file_id")
            data.entity_texts = json_data.get("entity_texts", [])
            data.relation_texts = json_data.get("relation_texts", [])
            
            return data
            
        except Exception as e:
            print(f"Failed to convert data: {e}")
            return None
    
    @staticmethod
    def save_pytorch_graph(data: Data, output_file: Path) -> bool:
        """Save PyTorch Geometric graph file"""
        try:
            torch.save(data, output_file)
            return True
        except Exception as e:
            print(f"Failed to save file {output_file}: {e}")
            return False
    
    def process_graphs(
        self,
        input_paths,
        output_dir: str,
        pattern: str = "graph_*.json",
        workers: int = 0,
    ) -> None:
        """Process all graph files"""
        if isinstance(input_paths, (str, Path)):
            candidates = [input_paths]
        else:
            candidates = list(input_paths)
        
        output_path = Path(output_dir)
        collected: List[Path] = []
        
        for raw_path in candidates:
            path = Path(raw_path)
            if not path.exists():
                print(f"Warning: Input path {path} does not exist, skipping")
                continue
            if path.is_file() and path.suffix.lower() == ".json":
                collected.append(path)
                continue
            if not path.is_dir():
                print(f"Warning: Input path {path} is not a directory, skipping")
                continue
            matched = list(path.glob(pattern))
            if not matched and pattern != "*.json":
                matched = list(path.glob("*.json"))
            if not matched:
                print(f"Info: No matching JSON files found in {path}")
                continue
            collected.extend(sorted(p for p in matched if p.is_file()))
        
        # Remove duplicates while preserving order
        seen = set()
        json_files: List[Path] = []
        for json_path in collected:
            if json_path in seen:
                continue
            seen.add(json_path)
            json_files.append(json_path)
        
        print(f"Found {len(json_files)} JSON files in total")
        if not json_files:
            print("No JSON files to process")
            return
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting to process graph files...")
        success_count = 0
        failed_count = 0
        
        if workers is None or workers < 0:
            workers = 0
        if workers == 0:
            worker_count = max(1, os.cpu_count() or 1)
        else:
            worker_count = workers
        worker_count = min(worker_count, len(json_files))
        
        if worker_count <= 1:
            for json_file in tqdm(json_files, desc="Converting graph files"):
                try:
                    json_data = self.load_json_graph(json_file)
                    if json_data is None:
                        failed_count += 1
                        continue
                    
                    pytorch_data = self.json_to_pytorch_data(json_data)
                    if pytorch_data is None:
                        failed_count += 1
                        continue
                    
                    output_file = output_path / f"{json_file.stem}.pt"
                    
                    if self.save_pytorch_graph(pytorch_data, output_file):
                        success_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    print(f"Error processing file {json_file}: {e}")
                    failed_count += 1
                    continue
        else:
            print(f"Using parallel processing: {worker_count} processes")
            chunk_size = max(1, len(json_files) // (worker_count * 4))
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
                results = executor.map(
                    _convert_single_graph,
                    (str(p) for p in json_files),
                    repeat(str(output_path)),
                    chunksize=chunk_size,
                )
                for success, message in tqdm(results, total=len(json_files), desc="Converting graph files"):
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        if message:
                            print(message)
        
        print(f"\nProcessing complete!")
        print(f"Success: {success_count} files")
        print(f"Failed: {failed_count} files")
        print(f"Output directory: {output_dir}")


def _convert_single_graph(json_path: str, output_dir: str) -> Tuple[bool, str]:
    json_file = Path(json_path)
    output_path = Path(output_dir)
    try:
        json_data = PyTorchGraphBuilder.load_json_graph(json_file)
        if json_data is None:
            return False, f"Failed to load file {json_file}"
        pytorch_data = PyTorchGraphBuilder.json_to_pytorch_data(json_data)
        if pytorch_data is None:
            return False, f"Failed to convert {json_file}"
        output_file = output_path / f"{json_file.stem}.pt"
        if PyTorchGraphBuilder.save_pytorch_graph(pytorch_data, output_file):
            return True, ""
        return False, f"Failed to save file {output_file}"
    except Exception as e:
        return False, f"Error processing file {json_file}: {e}"


def verify_pytorch_graphs(output_dir: str, num_samples: int = 3) -> None:
    """Verify generated PyTorch graph files"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory {output_path} does not exist")
        return
    
    # Get all .pt files
    pt_files = list(output_path.glob("graph_*.pt"))
    print(f"Found {len(pt_files)} .pt files")
    
    if not pt_files:
        print("No .pt files found")
        return
    
    # Verify first few files
    for i, pt_file in enumerate(pt_files[:num_samples]):
        print(f"\nVerifying file: {pt_file.name}")
        
        try:
            # Load PyTorch data (set weights_only=False to support PyTorch Geometric objects)
            try:
                data = torch.load(pt_file, weights_only=False)
            except TypeError:
                data = torch.load(pt_file)
            
            print(f"  Node feature shape: {data.x.shape}")
            print(f"  Edge index shape: {data.edge_index.shape}")
            print(f"  Edge feature shape: {data.edge_attr.shape}")
            print(f"  Graph feature shape: {data.graph_attr.shape}")
            print(f"  Number of nodes: {data.num_nodes}")
            print(f"  Number of edges: {data.num_edges}")
            
            # Check data types
            print(f"  Node feature type: {data.x.dtype}")
            print(f"  Edge index type: {data.edge_index.dtype}")
            print(f"  Edge feature type: {data.edge_attr.dtype}")
            print(f"  Graph feature type: {data.graph_attr.dtype}")
            
            # Check feature ranges
            print(f"  Node feature range: [{data.x.min():.4f}, {data.x.max():.4f}]")
            print(f"  Edge feature range: [{data.edge_attr.min():.4f}, {data.edge_attr.max():.4f}]")
            print(f"  Graph feature range: [{data.graph_attr.min():.4f}, {data.graph_attr.max():.4f}]")
            
            # Check summary
            if hasattr(data, 'summary'):
                print(f"  Summary length: {len(data.summary)}")
                print(f"  Summary preview: {data.summary[:100]}...")
            
            if hasattr(data, 'summary_embedding'):
                print(f"  Summary embedding shape: {data.summary_embedding.shape}")
                print(f"  Summary embedding range: [{data.summary_embedding.min():.4f}, {data.summary_embedding.max():.4f}]")
            
            print(f"  ✓ File verification passed")
            
        except Exception as e:
            print(f"  ✗ File verification failed: {e}")
    
    print(f"\n✓ Verification complete!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Convert scene graph JSON to PyTorch Geometric data")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["output/stage_03_scene_graphs"],
        help="Input JSON files or directories, can provide multiple paths",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/stage_04_pytorch",
        help="Output .pt file directory",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="graph_*.json",
        help="Glob pattern for matching JSON files, default only processes files with graph_ prefix",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel processing processes, 0 means auto-select based on CPU cores",
    )
    args = parser.parse_args()
    
    # Create graph builder
    try:
        builder = PyTorchGraphBuilder()
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install torch_geometric: pip install torch_geometric")
        return
    
    # Process graph files
    builder.process_graphs(
        args.inputs,
        args.output_dir,
        pattern=args.pattern,
        workers=args.workers,
    )
    
    # Verify results
    verify_pytorch_graphs(args.output_dir)


if __name__ == "__main__":
    main()
