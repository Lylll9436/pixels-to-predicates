#!/usr/bin/env python3
"""
Perception reasoning script based on GraphMAE and Bradley-Terry models
Performs six-dimensional perception prediction on 200 groundtruth images
"""

import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, multilabel_confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import GraphMAE related modules
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Perception dimensions
PERCEPTION_DIMENSIONS = ["safety", "lively", "boring", "wealthy", "depressing", "beautiful"]

# Path configuration
GROUNDTRUTH_IMAGES_DIR = Path("output/groundtruth/images")
GROUNDTRUTH_ANNOTATIONS_DIR = Path("output/groundtruth/annotations")
GROUNDTRUTH_SCENE_DIR = Path("output/groundtruth/scene")
GROUNDTRUTH_SCENE_GRAPHS_DIR = Path("output/groundtruth/scene_graphs")
GROUNDTRUTH_SCENE_GRAPHS_PYTORCH_DIR = Path("output/groundtruth/scene_graphs_pytorch")
GROUNDTRUTH_TEST_DIR = Path("output/groundtruth/test")
GRAPH_MAE_MODEL_PATH = Path("packed/graph_mae_best.pth")
BRADLEY_TERRY_MODEL_PATH = Path("result/bradley_terry_comparison_best.pth")
OUTPUT_PREDICT_DIR = Path("output/predictions")
OUTPUT_EVALUATION_DIR = Path("output/evaluation")

# GraphMAE configuration
GRAPH_MAE_CONFIG = {
    'node_feature_dim': 384,
    'edge_feature_dim': 384,
    'graph_feature_dim': 384,
    'hidden_dim': 256,
    'latent_dim': 128,
    'num_layers': 3,
    'dropout': 0.1,
}


class PerceptionPredictor(nn.Module):
    """Perception predictor based on GraphMAE features - outputs absolute scores"""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [256, 128, 64],
        num_classes: int = 6,
        dropout: float = 0.2
    ):
        super(PerceptionPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Shared feature extractor
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
        
        self.shared_encoder = nn.Sequential(*layers)
        
        # Independent regression head for each perception dimension - outputs continuous scores between 0-1
        self.regressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Ensure output is between 0-1
            ) for _ in range(num_classes)
        ])
    
    def forward(self, x):
        """Forward pass - outputs absolute scores"""
        shared_features = self.shared_encoder(x)
        
        # Independent prediction for each dimension
        predictions = []
        for regressor in self.regressors:
            pred = regressor(shared_features)
            predictions.append(pred.squeeze())
        
        return torch.stack(predictions, dim=1)  # [batch_size, num_classes]


class GraphMAE(nn.Module):
    """GraphMAE model definition (copied from 05_graph_vae.py)"""
    
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
        self.device = device or "cpu"
        
        # Input projections
        self.node_in = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_in = nn.Linear(edge_feature_dim, hidden_dim)
        
        # Learnable mask tokens
        self.node_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.edge_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.node_mask_token, std=0.02)
        nn.init.normal_(self.edge_mask_token, std=0.02)
        
        # GINE layers
        from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool
        
        def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            gin_mlp = build_mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
            conv = GINEConv(gin_mlp, train_eps=True, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity())
        
        # Graph-level projector
        self.graph_projector = build_mlp(hidden_dim * 2, hidden_dim, latent_dim, dropout)
        
        self.to(self.device)
    
    def encode_nodes(self, x_h: torch.Tensor, edge_index: torch.Tensor, e_h: torch.Tensor) -> torch.Tensor:
        h = x_h
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index, e_h)
            out = norm(out)
            out = F.relu(out)
            h = h + out  # residual
        return h
    
    def graph_readout(self, h: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        from torch_geometric.nn import global_max_pool, global_mean_pool
        mean_pooled = global_mean_pool(h, batch_index)
        max_pooled = global_max_pool(h, batch_index)
        return torch.cat([mean_pooled, max_pooled], dim=-1)
    
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


def load_graph_mae_model(model_path: Path, config: Dict, device: str = "cpu") -> GraphMAE:
    """Load GraphMAE model"""
    model = GraphMAE(
        node_feature_dim=config['node_feature_dim'],
        edge_feature_dim=config['edge_feature_dim'],
        graph_feature_dim=config['graph_feature_dim'],
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        device=device
    )
    
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        # Filter out decoder-related weights, keep only encoder part
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # Keep only encoder-related weights
            if not any(decoder_key in key for decoder_key in ['node_decoder', 'edge_decoder', 'graph_pred_head']):
                filtered_state_dict[key] = value
        
        # Load filtered weights
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠ Missing weights: {missing_keys}")
        if unexpected_keys:
            print(f"⚠ Unexpected weights: {unexpected_keys}")
        
        print(f"✓ Loaded GraphMAE model: {model_path}")
        print(f"✓ Successfully loaded {len(filtered_state_dict)} weight parameters")
    else:
        print(f"⚠ GraphMAE model file does not exist: {model_path}")
    
    model.eval()
    return model


def find_scene_graph(image_path: Path, scene_graphs_dir: Path) -> Optional[Path]:
    """Find corresponding scene graph file"""
    # Extract ID from image filename
    image_name = image_path.name
    if '_' in image_name:
        # Format: location_original_id.jpg
        original_id = image_name.split('_', 1)[1].rsplit('.', 1)[0]
    else:
        # Format: original_id.jpg
        original_id = image_path.stem
    
    # Find corresponding graph file
    graph_file = scene_graphs_dir / f"graph_{original_id}.pt"
    if graph_file.exists():
        return graph_file
    
    for pattern in ["*.pt"]:
        for graph_file in scene_graphs_dir.glob(pattern):
            if original_id in graph_file.name:
                return graph_file
    
    return None


def find_scene_description(image_path: Path, scene_dir: Path) -> Optional[Path]:
    """Find corresponding scene description file"""
    # Extract ID from image filename
    image_name = image_path.name
    if '_' in image_name:
        # Format: location_original_id.jpg
        original_id = image_name.split('_', 1)[1].rsplit('.', 1)[0]
        location = image_name.split('_', 1)[0]
        scene_file = scene_dir / f"{location}_{original_id}.json"
    else:
        # Format: original_id.jpg
        original_id = image_path.stem
        scene_file = scene_dir / f"{original_id}.json"
    
    if scene_file.exists():
        return scene_file
    
    return None


def batch_generate_scene_graphs_from_descriptions(missing_images: List[Dict], scene_dir: Path, scene_graphs_dir: Path) -> None:
    """Batch generate scene graphs from existing scene descriptions"""
    if not missing_images:
        return
    
    print(f"Starting batch processing of {len(missing_images)} images...")
    
    # 1. Load scene descriptions
    print("Step 1: Loading scene descriptions...")
    scene_descriptions = []
    for info in missing_images:
        scene_file = find_scene_description(info['path'], scene_dir)
        if scene_file and scene_file.exists():
            try:
                with open(scene_file, 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)
                scene_descriptions.append({
                    'info': info,
                    'scene_data': scene_data.get('scene_data', scene_data)  # Compatible with two formats
                })
            except Exception as e:
                print(f"✗ Failed to load scene description {info['name']}: {e}")
                continue
        else:
            print(f"✗ Scene description file does not exist: {info['name']}")
            continue
    
    if not scene_descriptions:
        print("✗ No scene descriptions found")
        return
    
    print(f"✓ Successfully loaded {len(scene_descriptions)} scene descriptions")
    
    # 2. Batch build scene graphs
    print("Step 2: Batch building scene graphs...")
    scene_graphs_data = batch_build_scene_graphs(scene_descriptions)
    
    if not scene_graphs_data:
        print("✗ No scene graphs built")
        return
    
    # 3. Batch convert to PyTorch format and save
    print("Step 3: Batch saving scene graphs...")
    batch_save_scene_graphs(scene_graphs_data, scene_graphs_dir)
    
    print(f"✓ Batch generation complete")


def batch_generate_scene_descriptions(missing_images: List[Dict]) -> List[Dict]:
    """Batch generate scene descriptions - using multiple API keys for parallel processing"""
    import os
    import base64
    import requests
    from queue import Queue, Empty
    from threading import Thread, Lock
    
    API_BASE = os.getenv("API_BASE", "https://chatapi.nloli.xyz")
    MODEL = os.getenv("MODEL", "gemini-2.5-flash")
    
    # Read API keys from environment variables
    api_keys_str = os.getenv("API_KEYS", "")
    if not api_keys_str:
        print("Warning: API_KEYS environment variable not set")
        return []
    API_KEYS = [key.strip() for key in api_keys_str.split(",") if key.strip()]
    PER_KEY_WORKERS = int(os.getenv("PER_KEY_WORKERS", "1"))
    
    if not missing_images:
        return []
    
    print(f"Using {len(API_KEYS)} API keys for parallel processing of {len(missing_images)} images")
    
    # Build prompt
    prompt_text = """You are an AI expert specializing in semantic scene analysis for autonomous driving systems.

Task
Analyze the provided street-view image and output a single JSON object describing the scene as semantic triplets. You must follow the procedure below and satisfy all hard constraints.

Hard constraints

Connectivity: No isolated entities; every entity must appear in ≥1 relation. If an entity cannot form a credible relation, omit it.

De-duplication & hierarchy: One entity ID per physical instance. Put brand/model/color and other fine-grained tags into that entity's attributes; never create alias entities. For groups, list members only if they have independent relations/actions and connect them via member_of.

Urban-perception priority: Prefer objects/states affecting perception (safety, liveliness, beauty, depression, boredom, wealth), e.g., graffiti, litter, street_light, street_tree/greenery, crosswalk_marking, bench, bus_stop, cleanliness, lighting_condition.

Open predicates: No fixed list. During reasoning, normalize predicates and resolve conflicts; only normalized predicates appear in the final JSON.

Attribute richness (anti-homogenization):

Each entity must include ≥2 non-trivial attributes (informative/discriminative; empty/placeholder values do not count).

Attribute diversity: For same-class entities, vary attribute dimensions/values (state/geometry/context) to avoid templating.

Fallback derived attributes (use only if needed to reach ≥2): occlusion_level (none/partial/heavy), bbox_area_ratio (0–1), distance_band (near/mid/far), visibility (clear/blurred/low_light/backlit).

Do not count duplicate/synonymous attributes; unknown/na does not count.

Language lock: All output must be in English. If inputs contain non-English text (e.g., signage), keep verbatim text in attributes as text_excerpt and optionally text_excerpt_native for the original script.

Attribute specification (to boost information and reduce homogenization)

Keys: lowercase snake_case; values are readable enums/phrases; numeric values include units when applicable (m, deg).

Recommended dimensions (cover ≥2 per entity; ≥3 encouraged):

Appearance/Material: color, material, type (sedan/suv/bench_with_backrest), brand/model.

State/Action: status (on/off/closed/damaged/under_construction), motion_state (moving/stopped/parked/waiting), signal_state (red/green/yellow/blinking)..

Environment/Conditions: lighting_condition (day/night/dusk/dawn), weather (sunny/cloudy/rain/snow), cleanliness (clean/littered), wear_level (new/moderate/worn).

Text/Semantics: text_excerpt (≤12 chars), iconography (pictogram_arrow/bus_symbol).

Group: count, density (sparse/medium/dense), spread (compact/clustered/line).

Greenery/Amenities: greenery_density (none/low/medium/high), canopy (present/absent), amenity_role (seating/shelter/bike_parking).

Merge attributes when multiple detections refer to the same object; do not create alias entities.

Reasoning & output procedure

Step 1: Reasoning (only inside <reasoning>; do not include it in the final output)

Entity recognition & normalization

Assign unique IDs (car_01, person_02, road_01…); use base classes for class. Put fine-grained tags into attributes (brand:"Toyota", type:"sedan", color:"white").

Enforce single-instance uniqueness (merge attributes across duplicates).

Groups vs members: only list members with independent actions/relations; add "【member】-【member_of】-【group】".

Use "stuff" anchors (road/sidewalk/facade/wall) to help connectivity.

Attribute check: ensure each kept entity has ≥2 non-trivial attributes; if not, add from fallback derived attributes; if still insufficient, remove the entity.

Relation identification (natural language → open predicates)

Enumerate key relations (spatial/topological/orientation/traffic action/signal state/functional interaction), then convert to predicates.

If a relation implies an entity state (e.g., stopped_at → motion_state:"stopped"), also write it back into attributes.

Predicate normalization (no preset list)

Form: lowercase snake_case lemmas (near, in_front_of, is_on, stopped_at, under_construction, waiting_for). Join phrases with underscores. If a phrase is non-English, translate to its English core phrase before normalizing.

Synonym convergence: cluster near-synonyms (next_to/adjacent/beside and non-English equivalents) → choose one canonical predicate (prefer the shortest unambiguous form, e.g., near) and replace all variants. Record the decision in <reasoning> only.

Conflict resolution: for the same (head, tail), keep only the higher-confidence relation (e.g., choose in_front_of over behind if evidence supports it). For strict containment (touching vs near), keep the stronger and drop the weaker.

Consistency: use the same canonical predicate string throughout this image.

Connectivity & quality check

Ensure every entity appears in ≥1 relation; otherwise connect via is_on to an anchor or near to a reasonable neighbor; if impossible, remove the entity.

Re-check attribute richness and de-homogenization for same-class entities.

Remove contradictory or duplicate relations.

Step 2: Final output (JSON only; no extra text)

You MUST return a single valid JSON object with EXACTLY these three top-level fields and nothing else:

"image_summary": 2–4 English sentences covering scene type; time/lighting/weather; key object counts and spatial layout (left/right/front/back/near/far); urban-perception cues (graffiti/litter/greenery density/lighting status/road markings/sign highlights); salient behaviors/traffic states (stopping/waiting/crossing/signal color). Avoid templated short sentences.

"entities": array of objects, each with id, class, attributes (≥2 non-trivial attributes; use fallback attributes if needed).

"semantic_triplets": array of strings, each strictly "【head】-【predicate】-【tail】". This array MUST be non-empty.

Strict completeness checks BEFORE returning:

The JSON is RFC 8259 compliant; no markdown, comments, or extra wrapper keys.

"semantic_triplets" exists and is non-empty.

All narrative strings are English (except verbatim text captured in text_excerpt/text_excerpt_native).

Hints for structured inputs (if the user supplies a prose/JSON description instead of an image)

Synthesize entities from the description (e.g., named buildings/signs/vehicles/people/traffic devices/road furniture).

Promote any named places or signs to entities with attributes including text_excerpt.

Ensure every entity participates in ≥1 relation even without pixel evidence (use is_on/near/along/in_front_of/behind/under_construction as appropriate).

If counts are given coarsely (e.g., "many"), create groups with count/density and relate members only when they have distinct actions.

Example (not a fixed vocabulary) <EXAMPLE>
Input description: graffiti on a corner wall, an overflowing trash bin by the curb, several pedestrians waiting under a street light for a red signal, street trees on both sides, and an office building adjacent to the sidewalk. <reasoning>

Candidate predicates: adjacent/near/on/wait for/show color/along…

Normalize: adjacent,next_to → near; wait for → waiting_for; on → is_on; show color → is_showing_color; Chinese phrases → English cores.

Attribute filling: add material:brick, condition:worn to wall_01; status:on, lighting_condition:night to street_light_01; status:overflow, relative_position:middle_right to trash_bin_01; use distance_band/visibility to separate similar entities.

</reasoning>
{
"image_summary": "At a nighttime intersection, street lights are on and the ground appears slightly wet; medium-coverage graffiti is visible on a side wall. Three pedestrians wait at a red signal, an overflowing trash bin sits near the sidewalk edge, and continuous street trees line both sides. An office façade abuts the sidewalk; greenery is moderate and overall cleanliness is average.",
"entities": [
{"id":"group_people_01","class":"group_people","attributes":{"count":3,"density":"compact"}},
{"id":"person_01","class":"person","attributes":{"occlusion_level":"partial","relative_position":"bottom_left"}},
{"id":"sidewalk_01","class":"sidewalk","attributes":{"material":"concrete","cleanliness":"littered"}},
{"id":"road_01","class":"road","attributes":{"material":"asphalt","lane_marking":"visible"}},
{"id":"traffic_light_01","class":"traffic_light","attributes":{"signal_state":"red","relative_position":"top_center"}},
{"id":"street_light_01","class":"street_light","attributes":{"status":"on","lighting_condition":"night"}},
],
"semantic_triplets": [
"【person_01】-【member_of】-【group_people_01】",
"【person_01】-【is_on】-【sidewalk_01】",
"【group_people_01】-【is_on】-【sidewalk_01】",
"【street_tree_01】-【along】-【road_01】",
"【building_01】-【near】-【sidewalk_01】"
]
}
</EXAMPLE>
---
"""

    # Create task queue
    task_queue: "Queue[Dict]" = Queue()
    for info in missing_images:
        task_queue.put(info)
    
    # Result storage
    scene_descriptions = []
    results_lock = Lock()
    success_count = 0
    failed_count = 0
    
    def worker(api_key: str, worker_name: str) -> None:
        nonlocal success_count, failed_count
        endpoint = f"{API_BASE}/v1beta/models/{MODEL}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}
        
        while True:
            try:
                info: Dict = task_queue.get(timeout=1)
            except Empty:
                break
            
            try:
                # Encode image
                with open(info['path'], "rb") as f:
                    image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")
                
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": prompt_text},
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": image_base64,
                                    }
                                },
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.1,
                        "topP": 0.9,
                        "topK": 64,
                        "responseMimeType": "application/json",
                    }
                }
                
                response = requests.post(
                    endpoint,
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=60,
                    verify=False
                )
                
                if response.status_code == 200:
                    result = response.json()
                    candidates = result.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        for part in parts:
                            if "text" in part:
                                text_content = part['text'].strip()
                                if text_content.startswith('```json'):
                                    text_content = text_content[7:]
                                    if text_content.endswith('```'):
                                        text_content = text_content[:-3]
                                    text_content = text_content.strip()
                                elif text_content.startswith('```'):
                                    text_content = text_content[3:]
                                    if text_content.endswith('```'):
                                        text_content = text_content[:-3]
                                    text_content = text_content.strip()
                                
                                try:
                                    scene_data = json.loads(text_content)
                                    with results_lock:
                                        scene_descriptions.append({
                                            'info': info,
                                            'scene_data': scene_data
                                        })
                                        success_count += 1
                                    break
                                except json.JSONDecodeError as e:
                                    print(f"✗ JSON parsing failed {info['name']}: {e}")
                                    break
                else:
                    print(f"✗ API request failed {info['name']}: {response.status_code}")
                    with results_lock:
                        failed_count += 1
                        
            except Exception as e:
                print(f"✗ Processing failed {info['name']}: {e}")
                with results_lock:
                    failed_count += 1
            finally:
                task_queue.task_done()
    
    # Start worker threads
    threads: List[Thread] = []
    for key_index, api_key in enumerate(API_KEYS, start=1):
        for worker_index in range(1, PER_KEY_WORKERS + 1):
            worker_name = f"Key{key_index}-T{worker_index}"
            thread = Thread(target=worker, args=(api_key, worker_name), daemon=True)
            thread.start()
            threads.append(thread)
    
    # 等待所有任务完成
    task_queue.join()
    
    for thread in threads:
        thread.join()
    
    print(f"✓ 并行处理完成: 成功 {success_count} 个，失败 {failed_count} 个")
    return scene_descriptions


def batch_build_scene_graphs(scene_descriptions: List[Dict]) -> List[Dict]:
    """批量构建场景图"""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # 初始化Sentence-BERT编码器
        print("初始化Sentence-BERT编码器...")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        scene_graphs_data = []
        
        for desc_info in tqdm(scene_descriptions, desc="构建场景图"):
            info = desc_info['info']
            scene_data = desc_info['scene_data']
            
            entities = scene_data.get("entities", [])
            triplets = scene_data.get("semantic_triplets", [])
            
            if not entities or not triplets:
                print(f"✗ 场景数据不完整 {info['name']}: entities={len(entities)}, triplets={len(triplets)}")
                continue
            
            # 编码实体
            entity_texts = []
            for entity in entities:
                class_name = entity["class"]
                attributes = entity.get("attributes", {})
                attr_text = " ".join([f"{k}:{v}" for k, v in attributes.items()])
                entity_text = f"{class_name} {attr_text}".strip()
                entity_texts.append(entity_text)
            
            # 编码关系
            relation_texts = []
            for triplet in triplets:
                if "【" in triplet and "】" in triplet:
                    parts = triplet.split("】-【")
                    if len(parts) == 3:
                        head = parts[0].replace("【", "")
                        predicate = parts[1]
                        tail = parts[2].replace("】", "")
                        relation_text = f"{head} {predicate} {tail}"
                        relation_texts.append(relation_text)
            
            # 编码文本为向量
            entity_embeddings = encoder.encode(entity_texts)
            relation_embeddings = encoder.encode(relation_texts) if relation_texts else np.array([]).reshape(0, 384)
            
            # 构建边索引
            edge_index = []
            edge_features = []
            
            for i, triplet in enumerate(triplets):
                if "【" in triplet and "】" in triplet:
                    parts = triplet.split("】-【")
                    if len(parts) == 3:
                        head_id = parts[0].replace("【", "")
                        tail_id = parts[2].replace("】", "")
                        
                        # 查找实体索引
                        head_idx = None
                        tail_idx = None
                        for j, entity in enumerate(entities):
                            if entity["id"] == head_id:
                                head_idx = j
                            if entity["id"] == tail_id:
                                tail_idx = j
                        
                        if head_idx is not None and tail_idx is not None:
                            edge_index.append([head_idx, tail_idx])
                            if i < len(relation_embeddings):
                                edge_features.append(relation_embeddings[i])
            
            if not edge_index:
                print(f"✗ 无法构建边索引 {info['name']}")
                continue
            
            # 构建图级特征
            summary = scene_data.get("image_summary", "")
            summary_embedding = encoder.encode([summary])[0] if summary else np.zeros(384)
            
            # 构建图特征（使用实体和关系的平均）
            all_features = np.vstack([entity_embeddings, relation_embeddings]) if len(relation_embeddings) > 0 else entity_embeddings
            graph_features = np.mean(all_features, axis=0)
            
            scene_graph = {
                "file_id": info['original_id'],
                "num_nodes": len(entities),
                "num_edges": len(edge_index),
                "node_features": entity_embeddings.tolist(),
                "edge_index": edge_index,
                "edge_features": edge_features,
                "graph_features": graph_features.tolist(),
                "summary": summary,
                "summary_embedding": summary_embedding.tolist(),
                "entity_texts": entity_texts,
                "relation_texts": relation_texts
            }
            
            scene_graphs_data.append(scene_graph)
        
        print(f"✓ 成功构建 {len(scene_graphs_data)} 个场景图")
        return scene_graphs_data
        
    except Exception as e:
        print(f"✗ 批量构建场景图失败: {e}")
        return []


def batch_save_scene_graphs(scene_graphs_data: List[Dict], scene_graphs_dir: Path) -> None:
    """批量保存场景图为PyTorch格式"""
    scene_graphs_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    for scene_graph_data in tqdm(scene_graphs_data, desc="保存场景图"):
        try:
            # 转换为PyTorch格式
            pytorch_graph = convert_to_pytorch_format(scene_graph_data)
            if pytorch_graph is None:
                continue
            
            # 保存图文件
            graph_file = scene_graphs_dir / f"graph_{scene_graph_data['file_id']}.pt"
            torch.save(pytorch_graph, graph_file)
            success_count += 1
            
        except Exception as e:
            print(f"✗ 保存场景图失败 {scene_graph_data['file_id']}: {e}")
            continue
    
    print(f"✓ 成功保存 {success_count} 个场景图")


def generate_scene_graph_if_missing(image_path: Path, scene_graphs_dir: Path) -> Optional[Path]:
    """如果场景图不存在，则生成它"""
    # 从图像文件名提取ID
    image_name = image_path.name
    if '_' in image_name:
        # 格式: location_original_id.jpg
        original_id = image_name.split('_', 1)[1].rsplit('.', 1)[0]
    else:
        # 格式: original_id.jpg
        original_id = image_path.stem
    
    # 检查是否已存在
    graph_file = scene_graphs_dir / f"graph_{original_id}.pt"
    if graph_file.exists():
        return graph_file
    
    print(f"⚠ 场景图不存在，开始生成: {original_id}")
    
    # 1. 使用Gemini API生成场景描述
    scene_description = generate_scene_description(image_path)
    if not scene_description:
        print(f"✗ 生成场景描述失败: {image_path.name}")
        return None
    
    # 2. 构建场景图
    scene_graph_data = build_scene_graph_from_description(scene_description, original_id)
    if not scene_graph_data:
        print(f"✗ 构建场景图失败: {image_path.name}")
        return None
    
    # 3. 转换为PyTorch格式并保存
    pytorch_graph = convert_to_pytorch_format(scene_graph_data)
    if pytorch_graph is None:
        print(f"✗ 转换为PyTorch格式失败: {image_path.name}")
        return None
    
    # 保存图文件
    try:
        scene_graphs_dir.mkdir(parents=True, exist_ok=True)
        torch.save(pytorch_graph, graph_file)
        print(f"✓ 场景图已生成: {graph_file}")
        return graph_file
    except Exception as e:
        print(f"✗ 保存场景图失败: {e}")
        return None


def generate_scene_description(image_path: Path) -> Optional[Dict]:
    """使用Gemini API生成场景描述"""
    import base64
    import requests
    
    import os
    API_BASE = os.getenv("API_BASE", "https://chatapi.nloli.xyz")
    MODEL = os.getenv("MODEL", "gemini-2.5-flash")
    
    # 从环境变量读取API密钥（使用第一个）
    api_keys_str = os.getenv("API_KEYS", "")
    if not api_keys_str:
        raise ValueError("未设置API_KEYS环境变量，请设置后再使用")
    API_KEY = api_keys_str.split(",")[0].strip()
    
    # 编码图片
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        print(f"✗ 图片编码失败: {e}")
        return None
    
    # 构建请求
    endpoint = f"{API_BASE}/v1beta/models/{MODEL}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}

    prompt_text = """You are an AI expert specializing in semantic scene analysis for autonomous driving systems.

Task
Analyze the provided street-view image and output a single JSON object describing the scene as semantic triplets. You must follow the procedure below and satisfy all hard constraints.

Hard constraints

Connectivity: No isolated entities; every entity must appear in ≥1 relation. If an entity cannot form a credible relation, omit it.

De-duplication & hierarchy: One entity ID per physical instance. Put brand/model/color and other fine-grained tags into that entity's attributes; never create alias entities. For groups, list members only if they have independent relations/actions and connect them via member_of.

Urban-perception priority: Prefer objects/states affecting perception (safety, liveliness, beauty, depression, boredom, wealth), e.g., graffiti, litter, street_light, street_tree/greenery, crosswalk_marking, bench, bus_stop, cleanliness, lighting_condition.

Open predicates: No fixed list. During reasoning, normalize predicates and resolve conflicts; only normalized predicates appear in the final JSON.

Attribute richness (anti-homogenization):

Each entity must include ≥2 non-trivial attributes (informative/discriminative; empty/placeholder values do not count).

Attribute diversity: For same-class entities, vary attribute dimensions/values (state/geometry/context) to avoid templating.

Fallback derived attributes (use only if needed to reach ≥2): occlusion_level (none/partial/heavy), bbox_area_ratio (0–1), distance_band (near/mid/far), visibility (clear/blurred/low_light/backlit).

Do not count duplicate/synonymous attributes; unknown/na does not count.

Language lock: All output must be in English. If inputs contain non-English text (e.g., signage), keep verbatim text in attributes as text_excerpt and optionally text_excerpt_native for the original script.

Attribute specification (to boost information and reduce homogenization)

Keys: lowercase snake_case; values are readable enums/phrases; numeric values include units when applicable (m, deg).

Recommended dimensions (cover ≥2 per entity; ≥3 encouraged):

Appearance/Material: color, material, type (sedan/suv/bench_with_backrest), brand/model.

State/Action: status (on/off/closed/damaged/under_construction), motion_state (moving/stopped/parked/waiting), signal_state (red/green/yellow/blinking)..

Environment/Conditions: lighting_condition (day/night/dusk/dawn), weather (sunny/cloudy/rain/snow), cleanliness (clean/littered), wear_level (new/moderate/worn).

Text/Semantics: text_excerpt (≤12 chars), iconography (pictogram_arrow/bus_symbol).

Group: count, density (sparse/medium/dense), spread (compact/clustered/line).

Greenery/Amenities: greenery_density (none/low/medium/high), canopy (present/absent), amenity_role (seating/shelter/bike_parking).

Merge attributes when multiple detections refer to the same object; do not create alias entities.

Reasoning & output procedure

Step 1: Reasoning (only inside <reasoning>; do not include it in the final output)

Entity recognition & normalization

Assign unique IDs (car_01, person_02, road_01…); use base classes for class. Put fine-grained tags into attributes (brand:"Toyota", type:"sedan", color:"white").

Enforce single-instance uniqueness (merge attributes across duplicates).

Groups vs members: only list members with independent actions/relations; add "【member】-【member_of】-【group】".

Use "stuff" anchors (road/sidewalk/facade/wall) to help connectivity.

Attribute check: ensure each kept entity has ≥2 non-trivial attributes; if not, add from fallback derived attributes; if still insufficient, remove the entity.

Relation identification (natural language → open predicates)

Enumerate key relations (spatial/topological/orientation/traffic action/signal state/functional interaction), then convert to predicates.

If a relation implies an entity state (e.g., stopped_at → motion_state:"stopped"), also write it back into attributes.

Predicate normalization (no preset list)

Form: lowercase snake_case lemmas (near, in_front_of, is_on, stopped_at, under_construction, waiting_for). Join phrases with underscores. If a phrase is non-English, translate to its English core phrase before normalizing.

Synonym convergence: cluster near-synonyms (next_to/adjacent/beside and non-English equivalents) → choose one canonical predicate (prefer the shortest unambiguous form, e.g., near) and replace all variants. Record the decision in <reasoning> only.

Conflict resolution: for the same (head, tail), keep only the higher-confidence relation (e.g., choose in_front_of over behind if evidence supports it). For strict containment (touching vs near), keep the stronger and drop the weaker.

Consistency: use the same canonical predicate string throughout this image.

Connectivity & quality check

Ensure every entity appears in ≥1 relation; otherwise connect via is_on to an anchor or near to a reasonable neighbor; if impossible, remove the entity.

Re-check attribute richness and de-homogenization for same-class entities.

Remove contradictory or duplicate relations.

Step 2: Final output (JSON only; no extra text)

You MUST return a single valid JSON object with EXACTLY these three top-level fields and nothing else:

"image_summary": 2–4 English sentences covering scene type; time/lighting/weather; key object counts and spatial layout (left/right/front/back/near/far); urban-perception cues (graffiti/litter/greenery density/lighting status/road markings/sign highlights); salient behaviors/traffic states (stopping/waiting/crossing/signal color). Avoid templated short sentences.

"entities": array of objects, each with id, class, attributes (≥2 non-trivial attributes; use fallback attributes if needed).

"semantic_triplets": array of strings, each strictly "【head】-【predicate】-【tail】". This array MUST be non-empty.

Strict completeness checks BEFORE returning:

The JSON is RFC 8259 compliant; no markdown, comments, or extra wrapper keys.

"semantic_triplets" exists and is non-empty.

All narrative strings are English (except verbatim text captured in text_excerpt/text_excerpt_native).

Hints for structured inputs (if the user supplies a prose/JSON description instead of an image)

Synthesize entities from the description (e.g., named buildings/signs/vehicles/people/traffic devices/road furniture).

Promote any named places or signs to entities with attributes including text_excerpt.

Ensure every entity participates in ≥1 relation even without pixel evidence (use is_on/near/along/in_front_of/behind/under_construction as appropriate).

If counts are given coarsely (e.g., "many"), create groups with count/density and relate members only when they have distinct actions.

Example (not a fixed vocabulary) <EXAMPLE>
Input description: graffiti on a corner wall, an overflowing trash bin by the curb, several pedestrians waiting under a street light for a red signal, street trees on both sides, and an office building adjacent to the sidewalk. <reasoning>

Candidate predicates: adjacent/near/on/wait for/show color/along…

Normalize: adjacent,next_to → near; wait for → waiting_for; on → is_on; show color → is_showing_color; Chinese phrases → English cores.

Attribute filling: add material:brick, condition:worn to wall_01; status:on, lighting_condition:night to street_light_01; status:overflow, relative_position:middle_right to trash_bin_01; use distance_band/visibility to separate similar entities.

</reasoning>
{
"image_summary": "At a nighttime intersection, street lights are on and the ground appears slightly wet; medium-coverage graffiti is visible on a side wall. Three pedestrians wait at a red signal, an overflowing trash bin sits near the sidewalk edge, and continuous street trees line both sides. An office façade abuts the sidewalk; greenery is moderate and overall cleanliness is average.",
"entities": [
{"id":"group_people_01","class":"group_people","attributes":{"count":3,"density":"compact"}},
{"id":"person_01","class":"person","attributes":{"occlusion_level":"partial","relative_position":"bottom_left"}},
{"id":"sidewalk_01","class":"sidewalk","attributes":{"material":"concrete","cleanliness":"littered"}},
{"id":"road_01","class":"road","attributes":{"material":"asphalt","lane_marking":"visible"}},
{"id":"traffic_light_01","class":"traffic_light","attributes":{"signal_state":"red","relative_position":"top_center"}},
{"id":"street_light_01","class":"street_light","attributes":{"status":"on","lighting_condition":"night"}},
],
"semantic_triplets": [
"【person_01】-【member_of】-【group_people_01】",
"【person_01】-【is_on】-【sidewalk_01】",
"【group_people_01】-【is_on】-【sidewalk_01】",
"【street_tree_01】-【along】-【road_01】",
"【building_01】-【near】-【sidewalk_01】"
]
}
</EXAMPLE>
---
"""

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.9,
            "topK": 64,
            "responseMimeType": "application/json",
        }
    }
    
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            params=params,
            json=payload,
            timeout=60,
            verify=False
        )
        
        if response.status_code == 200:
            result = response.json()
            candidates = result.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        text_content = part['text'].strip()
                        if text_content.startswith('```json'):
                            text_content = text_content[7:]
                            if text_content.endswith('```'):
                                text_content = text_content[:-3]
                            text_content = text_content.strip()
                        elif text_content.startswith('```'):
                            text_content = text_content[3:]
                            if text_content.endswith('```'):
                                text_content = text_content[:-3]
                            text_content = text_content.strip()
                        
                        try:
                            scene_data = json.loads(text_content)
                            return scene_data
                        except json.JSONDecodeError as e:
                            print(f"✗ JSON解析失败: {e}")
                            return None
        else:
            print(f"✗ API请求失败: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ API请求异常: {e}")
        return None


def build_scene_graph_from_description(scene_data: Dict, file_id: str) -> Optional[Dict]:
    """从场景描述构建场景图"""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # 初始化Sentence-BERT编码器
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        entities = scene_data.get("entities", [])
        triplets = scene_data.get("semantic_triplets", [])
        
        if not entities or not triplets:
            print(f"✗ 场景数据不完整: entities={len(entities)}, triplets={len(triplets)}")
            return None
        
        # 构建实体映射
        entity_map = {entity["id"]: entity for entity in entities}
        
        # 编码实体
        entity_texts = []
        for entity in entities:
            # 构建实体文本描述
            class_name = entity["class"]
            attributes = entity.get("attributes", {})
            attr_text = " ".join([f"{k}:{v}" for k, v in attributes.items()])
            entity_text = f"{class_name} {attr_text}".strip()
            entity_texts.append(entity_text)
        
        # 编码关系
        relation_texts = []
        for triplet in triplets:
            # 解析三元组
            if "【" in triplet and "】" in triplet:
                parts = triplet.split("】-【")
                if len(parts) == 3:
                    head = parts[0].replace("【", "")
                    predicate = parts[1]
                    tail = parts[2].replace("】", "")
                    relation_text = f"{head} {predicate} {tail}"
                    relation_texts.append(relation_text)
        
        # 编码文本为向量
        entity_embeddings = encoder.encode(entity_texts)
        relation_embeddings = encoder.encode(relation_texts) if relation_texts else np.array([]).reshape(0, 384)
        
        # 构建边索引
        edge_index = []
        edge_features = []
        
        for i, triplet in enumerate(triplets):
            if "【" in triplet and "】" in triplet:
                parts = triplet.split("】-【")
                if len(parts) == 3:
                    head_id = parts[0].replace("【", "")
                    tail_id = parts[2].replace("】", "")
                    
                    # 查找实体索引
                    head_idx = None
                    tail_idx = None
                    for j, entity in enumerate(entities):
                        if entity["id"] == head_id:
                            head_idx = j
                        if entity["id"] == tail_id:
                            tail_idx = j
                    
                    if head_idx is not None and tail_idx is not None:
                        edge_index.append([head_idx, tail_idx])
                        if i < len(relation_embeddings):
                            edge_features.append(relation_embeddings[i])
        
        if not edge_index:
            print(f"✗ 无法构建边索引")
            return None
        
        # 构建图级特征
        summary = scene_data.get("image_summary", "")
        summary_embedding = encoder.encode([summary])[0] if summary else np.zeros(384)
        
        # 构建图特征（使用实体和关系的平均）
        all_features = np.vstack([entity_embeddings, relation_embeddings]) if len(relation_embeddings) > 0 else entity_embeddings
        graph_features = np.mean(all_features, axis=0)
        
        scene_graph = {
            "file_id": file_id,
            "num_nodes": len(entities),
            "num_edges": len(edge_index),
            "node_features": entity_embeddings.tolist(),
            "edge_index": edge_index,
            "edge_features": edge_features,
            "graph_features": graph_features.tolist(),
            "summary": summary,
            "summary_embedding": summary_embedding.tolist(),
            "entity_texts": entity_texts,
            "relation_texts": relation_texts
        }
        
        return scene_graph
        
    except Exception as e:
        print(f"✗ 构建场景图失败: {e}")
        return None


def convert_to_pytorch_format(scene_graph_data: Dict) -> Optional[Data]:
    """将场景图数据转换为PyTorch Geometric格式"""
    try:
        # 提取特征
        node_features = torch.tensor(scene_graph_data["node_features"], dtype=torch.float32)
        edge_index = torch.tensor(scene_graph_data["edge_index"], dtype=torch.long)
        edge_features = torch.tensor(scene_graph_data["edge_features"], dtype=torch.float32)
        graph_features = torch.tensor(scene_graph_data["graph_features"], dtype=torch.float32)
        
        # 确保edge_index的形状正确
        if edge_index.shape[0] != 2:
            edge_index = edge_index.T
        
        # 创建PyTorch Geometric Data对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            graph_attr=graph_features,
            num_nodes=scene_graph_data["num_nodes"],
            num_edges=scene_graph_data["num_edges"]
        )
        
        # 添加额外信息
        data.summary = scene_graph_data.get("summary", "")
        summary_embedding = scene_graph_data.get("summary_embedding", [])
        data.summary_embedding = torch.tensor(summary_embedding, dtype=torch.float32) if summary_embedding else torch.empty(0, dtype=torch.float32)
        data.file_id = scene_graph_data.get("file_id")
        data.entity_texts = scene_graph_data.get("entity_texts", [])
        data.relation_texts = scene_graph_data.get("relation_texts", [])
        
        return data
        
    except Exception as e:
        print(f"✗ 转换为PyTorch格式失败: {e}")
        return None


def load_scene_graph(graph_path: Path) -> Optional[Data]:
    """加载场景图数据"""
    try:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        
        if isinstance(graph, dict):
            graph = Data(**graph)
        
        if not isinstance(graph, Data):
            print(f"⚠ 无效的图数据格式: {graph_path}")
            return None
        
        if graph.x is None or graph.x.numel() == 0:
            print(f"⚠ 空图数据: {graph_path}")
            return None
        
        return graph
        
    except Exception as e:
        print(f"✗ 加载图数据失败 {graph_path}: {e}")
        return None


def extract_graph_features(graph: Data, graph_mae_model: GraphMAE) -> torch.Tensor:
    """使用GraphMAE提取图特征"""
    # 确保所有张量在同一设备上
    device = next(graph_mae_model.parameters()).device
    graph = graph.to(device)
    
    # 添加batch维度
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    
    with torch.no_grad():
        features = graph_mae_model.encode_graph(graph)
    
    return features


def load_groundtruth_annotations(annotations_dir: Path) -> Dict[str, Dict]:
    """加载groundtruth标注 - 转换为绝对分数"""
    annotations = {}
    
    for json_file in annotations_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_id = data['image_id']
            perception_scores = data['perception_scores']
            
            # 将yes/no转换为绝对分数 (yes=1.0, no=0.0)
            # 在实际应用中，这些分数应该基于更细粒度的标注
            absolute_scores = {}
            for dim in PERCEPTION_DIMENSIONS:
                if perception_scores[dim] == "yes":
                    absolute_scores[dim] = 1.0
                else:
                    absolute_scores[dim] = 0.0
            
            annotations[image_id] = {
                'location': data['location'],
                'scores': absolute_scores,
                'raw_scores': perception_scores
            }
            
        except Exception as e:
            print(f"✗ 加载标注失败 {json_file}: {e}")
            continue
    
    print(f"✓ 加载了 {len(annotations)} 个groundtruth标注")
    return annotations


def create_perception_predictor(input_dim: int = 128) -> PerceptionPredictor:
    """创建感知预测器"""
    model = PerceptionPredictor(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        num_classes=len(PERCEPTION_DIMENSIONS),
        dropout=0.2
    )
    
    # 初始化权重
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    return model


def train_perception_predictor(
    model: PerceptionPredictor,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cpu",
    epochs: int = 100,
    learning_rate: float = 1e-3
) -> PerceptionPredictor:
    """训练感知预测器 - 回归任务"""
    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()  # 使用MSE损失进行回归
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        predictions = model(features)
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    model.eval()
    return model


def train_perception_predictor_with_validation(
    model: PerceptionPredictor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: str = "cpu",
    epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 10
) -> PerceptionPredictor:
    """带验证的训练感知预测器 - 回归任务"""
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()  # 使用MSE损失进行回归
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        
        train_predictions = model(X_train)
        train_loss = criterion(train_predictions, y_train)
        
        train_loss.backward()
        optimizer.step()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    return model


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dimensions: List[str]
) -> Dict:
    """评估预测结果 - 回归任务"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {}
    
    # 整体回归指标
    results['overall'] = {
        'mse': mean_squared_error(y_true.flatten(), y_pred.flatten()),
        'rmse': np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten())),
        'mae': mean_absolute_error(y_true.flatten(), y_pred.flatten()),
        'r2': r2_score(y_true.flatten(), y_pred.flatten()),
    }
    
    # 每个维度的指标
    results['per_dimension'] = {}
    for i, dim in enumerate(dimensions):
        results['per_dimension'][dim] = {
            'mse': mean_squared_error(y_true[:, i], y_pred[:, i]),
            'rmse': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
            'r2': r2_score(y_true[:, i], y_pred[:, i]),
            'correlation': np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        }
    
    return results


def plot_evaluation_results(results: Dict, output_dir: Path):
    """绘制评估结果 - 回归任务"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 整体性能图
    dimensions = list(results['per_dimension'].keys())
    metrics = ['rmse', 'mae', 'r2', 'correlation']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results['per_dimension'][dim][metric] for dim in dimensions]
        
        axes[i].bar(dimensions, values)
        axes[i].set_title(f'{metric.upper()} by Dimension')
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
        
        # 设置不同的y轴范围
        if metric in ['rmse', 'mae']:
            axes[i].set_ylim(0, max(values) * 1.1)
        else:  # r2, correlation
            axes[i].set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_predictions_table(
    image_ids: List[str],
    image_locations: List[str],
    predictions: np.ndarray,
    dimensions: List[str],
    output_dir: Path,
    split_lookup: Optional[Dict[int, str]] = None,
    filename_prefix: str = "all_predictions"
) -> Tuple[Path, Path]:
    """����ÿ��ͼ����֪ά�������ŵ�CSV/JSON��"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{filename_prefix}.csv"
    json_path = output_dir / f"{filename_prefix}.json"
    
    header = ["index", "image_id", "location"]
    if split_lookup is not None:
        header.append("split")
    header.extend(dimensions)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for idx, (image_id, location) in enumerate(zip(image_ids, image_locations)):
            row = [idx, image_id, location]
            if split_lookup is not None:
                row.append(split_lookup.get(idx, "unknown"))
            row.extend(f"{float(predictions[idx, dim_idx]):.6f}" for dim_idx in range(predictions.shape[1]))
            writer.writerow(row)
    
    records = []
    for idx, (image_id, location) in enumerate(zip(image_ids, image_locations)):
        record = {
            "index": idx,
            "image_id": image_id,
            "location": location,
            "scores": {
                dimension: float(predictions[idx, dim_idx])
                for dim_idx, dimension in enumerate(dimensions)
            }
        }
        if split_lookup is not None:
            record["split"] = split_lookup.get(idx, "unknown")
        records.append(record)
    
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(records, json_file, ensure_ascii=False, indent=2)
    
    return csv_path, json_path


def generate_perception_score_grid(
    image_paths: List[Path],
    predictions: np.ndarray,
    dimensions: List[str],
    output_dir: Path,
    grid_size: int = 4,
    max_images_per_dim: int = 16
) -> None:
    """生成类似Place Pulse风格的感知维度分数网格图"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个维度创建排序的图像
    for dim_idx, dimension in enumerate(dimensions):
        # 获取该维度的分数
        scores = predictions[:, dim_idx]
        
        # 排序图像索引
        sorted_indices = np.argsort(scores)
        
        # 选择分数最低和最高的图像
        low_score_indices = sorted_indices[:max_images_per_dim//2]
        high_score_indices = sorted_indices[-max_images_per_dim//2:]
        
        # 创建网格
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        # 填充低分数图像（左侧）
        for i, idx in enumerate(low_score_indices):
            if i < grid_size * grid_size // 2:
                try:
                    from PIL import Image
                    img = Image.open(image_paths[idx])
                    axes[i].imshow(img)
                    axes[i].set_title(f'Score: {scores[idx]:.3f}', fontsize=8)
                    axes[i].axis('off')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Error\n{image_paths[idx].name}', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
        
        # 填充高分数图像（右侧）
        for i, idx in enumerate(high_score_indices):
            grid_pos = i + grid_size * grid_size // 2
            if grid_pos < grid_size * grid_size:
                try:
                    from PIL import Image
                    img = Image.open(image_paths[idx])
                    axes[grid_pos].imshow(img)
                    axes[grid_pos].set_title(f'Score: {scores[idx]:.3f}', fontsize=8)
                    axes[grid_pos].axis('off')
                except Exception as e:
                    axes[grid_pos].text(0.5, 0.5, f'Error\n{image_paths[idx].name}', 
                                      ha='center', va='center', transform=axes[grid_pos].transAxes)
                    axes[grid_pos].axis('off')
        
        # 隐藏未使用的子图
        for i in range(len(image_paths), grid_size * grid_size):
            axes[i].axis('off')
        
        # 添加标题和标签
        fig.suptitle(f'{dimension.capitalize()} Dimension\nLow ← → High', fontsize=16, fontweight='bold')
        
        # 添加维度标签
        fig.text(0.02, 0.5, dimension.capitalize(), rotation=90, 
                fontsize=14, fontweight='bold', va='center')
        
        # 添加分数范围标签
        fig.text(0.5, 0.02, 'Low', ha='center', fontsize=12)
        fig.text(0.5, 0.98, 'High', ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{dimension}_score_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 生成 {dimension} 维度分数网格图")


def generate_all_images_score_grid(
    image_paths: List[Path],
    predictions: np.ndarray,
    dimensions: List[str],
    output_dir: Path,
    grid_size: int = 8
) -> None:
    """为所有图像生成感知维度分数网格图"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个维度创建排序
    for dim_idx, dimension in enumerate(dimensions):
        scores = predictions[:, dim_idx]
        sorted_indices = np.argsort(scores)
        
        # 计算网格大小
        num_images = len(image_paths)
        rows = int(np.ceil(np.sqrt(num_images)))
        cols = int(np.ceil(num_images / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]
        
        axes = np.array(axes).flatten()
        
        # 填充图像
        for i, idx in enumerate(sorted_indices):
            if i < len(axes):
                try:
                    from PIL import Image
                    img = Image.open(image_paths[idx])
                    axes[i].imshow(img)
                    axes[i].set_title(f'{scores[idx]:.3f}', fontsize=8)
                    axes[i].axis('off')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Error\n{image_paths[idx].name}', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
        
        # 隐藏未使用的子图
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f'{dimension.capitalize()} Dimension - All Images\n(Sorted by Score)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{dimension}_all_images_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 生成 {dimension} 维度所有图像分数网格图")


def main():
    """主函数"""
    print("=== 感知推理脚本 ===")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建输出目录
    OUTPUT_PREDICT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载GraphMAE模型
    print("\n1. 加载GraphMAE模型...")
    graph_mae_model = load_graph_mae_model(GRAPH_MAE_MODEL_PATH, GRAPH_MAE_CONFIG, device)
    
    # 2. 加载groundtruth标注
    print("\n2. 加载groundtruth标注...")
    groundtruth_annotations = load_groundtruth_annotations(GROUNDTRUTH_ANNOTATIONS_DIR)
    
    if not groundtruth_annotations:
        print("✗ 未找到groundtruth标注，退出")
        return
    
    # 3. 批量检查和处理场景图
    print("\n3. 批量检查和处理场景图...")
    
    # 收集所有图像信息
    image_info = []
    for image_path in list(GROUNDTRUTH_IMAGES_DIR.glob("*.jpg")) + list(GROUNDTRUTH_IMAGES_DIR.glob("*.jpeg")):
        image_name = image_path.name
        if '_' in image_name:
            location, original_id = image_name.split('_', 1)
            original_id = original_id.rsplit('.', 1)[0]
        else:
            location = "unknown"
            original_id = image_path.stem
        
        image_info.append({
            'path': image_path,
            'location': location,
            'original_id': original_id,
            'name': image_name
        })
    
    print(f"找到 {len(image_info)} 张图像")
    
    # 检查哪些场景图缺失
    missing_scene_graphs = []
    existing_scene_graphs = []
    
    for info in image_info:
        scene_graph_path = find_scene_graph(info['path'], GROUNDTRUTH_SCENE_GRAPHS_PYTORCH_DIR)
        if scene_graph_path:
            existing_scene_graphs.append((info, scene_graph_path))
        else:
            missing_scene_graphs.append(info)
    
    print(f"已存在场景图: {len(existing_scene_graphs)} 张")
    print(f"缺失场景图: {len(missing_scene_graphs)} 张")
    
    # 批量生成缺失的场景图（从现有场景描述）
    if missing_scene_graphs:
        print(f"\n开始批量生成 {len(missing_scene_graphs)} 个缺失的场景图...")
        batch_generate_scene_graphs_from_descriptions(missing_scene_graphs, GROUNDTRUTH_SCENE_DIR, GROUNDTRUTH_SCENE_GRAPHS_PYTORCH_DIR)
        
        # 重新检查生成结果
        newly_generated = []
        for info in missing_scene_graphs:
            scene_graph_path = find_scene_graph(info['path'], GROUNDTRUTH_SCENE_GRAPHS_PYTORCH_DIR)
            if scene_graph_path:
                newly_generated.append((info, scene_graph_path))
            else:
                print(f"✗ 生成失败: {info['name']}")
        
        existing_scene_graphs.extend(newly_generated)
        print(f"✓ 成功生成 {len(newly_generated)} 个场景图")
    
    # 4. 批量提取特征
    print(f"\n4. 批量提取特征...")
    image_features = []
    image_labels = []
    image_ids = []
    image_locations = []
    image_paths_all = []
    
    processed_count = 0
    failed_count = 0
    
    for info, scene_graph_path in tqdm(existing_scene_graphs, desc="提取特征"):
        # 加载场景图
        graph = load_scene_graph(scene_graph_path)
        if graph is None:
            failed_count += 1
            continue
        
        # 提取特征
        try:
            features = extract_graph_features(graph, graph_mae_model)
            image_features.append(features.cpu())
            image_ids.append(info['original_id'])
            image_locations.append(info['location'])
            image_paths_all.append(info['path'])
            
            # 获取标签
            if info['original_id'] in groundtruth_annotations:
                labels = [groundtruth_annotations[info['original_id']]['scores'][dim] for dim in PERCEPTION_DIMENSIONS]
                image_labels.append(labels)
            else:
                print(f"⚠ 未找到标注: {info['original_id']}")
                image_labels.append([0.0] * len(PERCEPTION_DIMENSIONS))  # 默认分数
            
            processed_count += 1
            
        except Exception as e:
            print(f"✗ 处理失败 {info['name']}: {e}")
            failed_count += 1
            continue
    
    print(f"✓ 成功处理 {processed_count} 张图像，失败 {failed_count} 张")
    
    if not image_features:
        print("✗ 没有成功处理的图像，退出")
        return
    
    # 4. 准备训练数据并划分数据集
    print("\n4. 准备训练数据并划分数据集...")
    X = torch.cat(image_features, dim=0)
    y = torch.tensor(image_labels, dtype=torch.float32)
    
    print(f"特征维度: {X.shape}")
    print(f"标签维度: {y.shape}")
    
    # 划分数据集：训练集60%，验证集20%，测试集20%
    X_np = X.numpy()
    y_np = y.numpy()
    
    # 首先划分训练集和临时集（80%训练+验证，20%测试）
    X_train_val, X_test, y_train_val, y_test, indices_train_val, indices_test = train_test_split(
        X_np, y_np, range(len(X_np)), test_size=0.2, random_state=42, stratify=None
    )
    
    # 再划分训练集和验证集（75%训练，25%验证）
    X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(
        X_train_val, y_train_val, indices_train_val, test_size=0.25, random_state=42, stratify=None
    )
    
    # 转换回PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 统一维护index索引
    indices_train = [int(i) for i in indices_train]
    indices_val = [int(i) for i in indices_val]
    indices_test = [int(i) for i in indices_test]
    
    # 获取对应的图像ID和位置信息
    train_ids = [image_ids[i] for i in indices_train]
    train_locations = [image_locations[i] for i in indices_train]
    val_ids = [image_ids[i] for i in indices_val]
    val_locations = [image_locations[i] for i in indices_val]
    test_ids = [image_ids[i] for i in indices_test]
    test_locations = [image_locations[i] for i in indices_test]
    
    # 构建索引到数据集划分的映射
    split_lookup = {idx: "train" for idx in indices_train}
    split_lookup.update({idx: "val" for idx in indices_val})
    split_lookup.update({idx: "test" for idx in indices_test})
    
    # 5. 创建和训练感知预测器
    print("\n5. 训练感知预测器...")
    predictor = create_perception_predictor(input_dim=X_train.shape[1])
    
    # 使用训练集训练，验证集调参
    predictor = train_perception_predictor_with_validation(
        predictor, X_train, y_train, X_val, y_val, device=device, epochs=100, learning_rate=1e-3
    )
    
    # 6. 对全部图像生成绝对分数并导出
    print("\n6. 对全部图像生成绝对分数并导出...")
    predictor.eval()
    with torch.no_grad():
        X_all_device = X.to(device)
        predictions_all = predictor(X_all_device).cpu().numpy()
    all_predictions_csv, all_predictions_json = save_predictions_table(
        image_ids,
        image_locations,
        predictions_all,
        PERCEPTION_DIMENSIONS,
        OUTPUT_PREDICT_DIR,
        split_lookup=split_lookup,
        filename_prefix="all_predictions"
    )
    print(f"✓ 所有图像预测结果CSV: {all_predictions_csv}")
    print(f"✓ 所有图像预测结果JSON: {all_predictions_json}")
    
    # 7. 在测试集上进行预测
    print("\n7. 在测试集上进行预测...")
    with torch.no_grad():
        X_test_device = X_test.to(device)
        predictions_scores_test = predictor(X_test_device).cpu().numpy()
    
    # 8. 保存预测结果（仅保存测试集结果）
    print("\n8. 保存预测结果...")
    for i, (image_id, location) in enumerate(zip(test_ids, test_locations)):
        result = {
            "image_id": image_id,
            "location": location,
            "perception_scores": {},
            "ground_truth": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for j, dim in enumerate(PERCEPTION_DIMENSIONS):
            result["perception_scores"][dim] = float(predictions_scores_test[i, j])
            result["ground_truth"][dim] = float(y_test[i, j])
        
        # 保存预测结果
        output_file = OUTPUT_PREDICT_DIR / f"{location}_{image_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 测试集预测结果已保存到: {OUTPUT_PREDICT_DIR}")
    
    # 9. 评估性能（仅在测试集上）
    print("\n9. 评估性能...")
    y_true = y_test.numpy()
    y_pred = predictions_scores_test
    
    evaluation_results = evaluate_predictions(y_true, y_pred, PERCEPTION_DIMENSIONS)
    
    # 保存评估结果
    with open(OUTPUT_EVALUATION_DIR / 'evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    # 绘制评估结果
    plot_evaluation_results(evaluation_results, OUTPUT_EVALUATION_DIR)
    
    # 打印评估结果
    print("\n=== 测试集评估结果 ===")
    print(f"测试集大小: {len(test_ids)} 样本")
    print(f"整体RMSE: {evaluation_results['overall']['rmse']:.4f}")
    print(f"整体MAE: {evaluation_results['overall']['mae']:.4f}")
    print(f"整体R²: {evaluation_results['overall']['r2']:.4f}")
    
    print("\n各维度性能:")
    for dim in PERCEPTION_DIMENSIONS:
        metrics = evaluation_results['per_dimension'][dim]
        print(f"{dim:12}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}, Corr={metrics['correlation']:.3f}")
    
    print(f"\n✓ 评估结果已保存到: {OUTPUT_EVALUATION_DIR}")
    
    # 10. 生成感知维度分数网格图
    print("\n10. 生成感知维度分数网格图...")
    
    # 全量图像排序展示
    if image_paths_all:
        generate_perception_score_grid(
            image_paths_all,
            predictions_all,
            PERCEPTION_DIMENSIONS,
            OUTPUT_EVALUATION_DIR / "score_grids",
            grid_size=4,
            max_images_per_dim=16
        )
        
        generate_all_images_score_grid(
            image_paths_all,
            predictions_all,
            PERCEPTION_DIMENSIONS,
            OUTPUT_EVALUATION_DIR / "all_images_grids"
        )
        
        print(f"✓ 全量图像分数网格图已保存到: {OUTPUT_EVALUATION_DIR / 'score_grids'}")
    else:
        print("⚠ 未找到全量图像路径，跳过全量网格图生成")
    
    # 测试集图像排序展示
    if indices_test:
        test_image_paths = [image_paths_all[i] for i in indices_test]
        
        generate_perception_score_grid(
            test_image_paths,
            predictions_scores_test,
            PERCEPTION_DIMENSIONS,
            OUTPUT_EVALUATION_DIR / "score_grids",
            grid_size=4,
            max_images_per_dim=16
        )
        
        generate_all_images_score_grid(
            test_image_paths,
            predictions_scores_test,
            PERCEPTION_DIMENSIONS,
            OUTPUT_EVALUATION_DIR / "all_images_grids"
        )
        
        print(f"✓ 测试集分数网格图已保存到: {OUTPUT_EVALUATION_DIR / 'score_grids'}")
    else:
        print("⚠ 测试集索引为空，跳过测试集网格图生成")
    

# 保存数据集划分信息
    dataset_split_info = {
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "test_samples": len(test_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "train_locations": train_locations,
        "val_locations": val_locations,
        "test_locations": test_locations,
        "random_seed": 42,
        "split_ratio": "60% train, 20% val, 20% test",
        "index_split_map": split_lookup
    }
    
    with open(OUTPUT_EVALUATION_DIR / 'dataset_split.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_split_info, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 数据集划分信息已保存到: {OUTPUT_EVALUATION_DIR / 'dataset_split.json'}")
    
    # 11. 创建测试集文件夹并复制测试集图像
    print("\n11. 创建测试集文件夹并复制测试集图像...")
    GROUNDTRUTH_TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    for i, (image_id, location) in enumerate(zip(test_ids, test_locations)):
        # 查找原始图像文件
        original_image_name = f"{location}_{image_id}.jpg"
        original_image_path = GROUNDTRUTH_IMAGES_DIR / original_image_name
        
        if original_image_path.exists():
            # 复制到测试集文件夹
            test_image_path = GROUNDTRUTH_TEST_DIR / original_image_name
            try:
                import shutil
                shutil.copy2(original_image_path, test_image_path)
                copied_count += 1
            except Exception as e:
                print(f"✗ 复制图像失败 {original_image_name}: {e}")
        else:
            print(f"⚠ 原始图像不存在: {original_image_name}")
    
    print(f"✓ 成功复制 {copied_count} 张测试集图像到: {GROUNDTRUTH_TEST_DIR}")


if __name__ == "__main__":
    main()
