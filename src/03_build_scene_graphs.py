#!/usr/bin/env python3
"""
Enhanced scene graph builder - Uses pp2_full.json data, preserves structural relationships and correctly aligns file IDs
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re
import torch
from tqdm import tqdm
import warnings

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: Unable to import sentence-transformers")


@dataclass
class Entity:
    """Entity class"""
    id: str
    class_name: str
    attributes: Dict


@dataclass
class Relation:
    """Relation class"""
    subject: str
    predicate: str
    object: str


class SentenceBERTEncoder:
    """Sentence-BERT encoder - Supports GPU acceleration"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """Initialize Sentence-BERT encoder
        
        Args:
            model_name: Model name
            device: Device type ('cuda', 'cpu', 'auto' or None)
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default dimension for all-MiniLM-L6-v2
        self.device = self._get_device(device)
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._init_model()
        else:
            print("Warning: sentence-transformers unavailable, will use random vectors")
    
    def _get_device(self, device: Optional[str]) -> str:
        """Get computing device"""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                device = "cuda"
                print(f"✓ CUDA detected, using GPU acceleration (Device: {torch.cuda.get_device_name()})")
            else:
                device = "cpu"
                print("⚠ CUDA unavailable, using CPU computation")
        elif device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA requested but unavailable, falling back to CPU")
            device = "cpu"
        
        return device
    
    def _init_model(self):
        """Initialize Sentence-BERT model"""
        try:
            print(f"Loading Sentence-BERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Set device
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                print(f"✓ Model loaded to GPU")
            else:
                print(f"✓ Model using CPU")
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully, embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"Warning: Unable to load Sentence-BERT model: {e}")
            self.model = None
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode text to vectors
        
        Args:
            texts: List of texts
            batch_size: Batch size, can use larger batches on GPU
            
        Returns:
            Encoded vector array
        """
        if self.model is not None:
            try:
                # Use larger batch size on GPU
                if self.device == "cuda":
                    batch_size = min(batch_size * 4, 128)  # GPU can handle larger batches
                
                embeddings = self.model.encode(
                    texts, 
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    show_progress_bar=len(texts) > 100  # Show progress bar for large text volumes
                )
                return embeddings
            except Exception as e:
                print(f"Warning: Sentence-BERT encoding failed: {e}")
                return self._encode_fallback(texts)
        else:
            return self._encode_fallback(texts)
    
    def _encode_fallback(self, texts: List[str]) -> np.ndarray:
        """Fallback encoding method"""
        # Use simple hash encoding as fallback
        embeddings = []
        for text in texts:
            # Simple hash encoding
            hash_val = hash(text) % (2**32)
            # Convert to fixed-dimensional vector
            vec = np.zeros(self.embedding_dim)
            for i in range(min(len(str(hash_val)), self.embedding_dim)):
                vec[i] = float(str(hash_val)[i]) if i < len(str(hash_val)) else 0.0
            embeddings.append(vec)
        return np.array(embeddings)


class EnhancedSceneGraphBuilder:
    """Enhanced scene graph builder - Supports GPU acceleration"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """Initialize scene graph builder
        
        Args:
            model_name: Sentence-BERT model name
            device: Computing device ('cuda', 'cpu', 'auto' or None)
        """
        self.device = device
        self.encoder = SentenceBERTEncoder(model_name, device)
        self.embedding_dim = self.encoder.embedding_dim
        
        # GPU memory management
        if self.encoder.device == "cuda":
            self._setup_gpu_memory()
    
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
    
    def parse_scene_data(self, file_id: str, scene_data: Dict) -> Tuple[List[Entity], List[Relation], str, Dict]:
        """Parse scene data"""
        entities = []
        relations = []
        
        # Parse entities
        if "entities" in scene_data:
            for entity_data in scene_data["entities"]:
                entity = Entity(
                    id=entity_data["id"],
                    class_name=entity_data["class"],
                    attributes=entity_data.get("attributes", {})
                )
                entities.append(entity)
        
        # Parse relations - handle new format
        if "semantic_triplets" in scene_data:
            semantic_triplets = scene_data["semantic_triplets"]
            if isinstance(semantic_triplets, list):
                for triplet_data in semantic_triplets:
                    relation = self._parse_triplet_new_format(triplet_data)
                    if relation:
                        relations.append(relation)
        
        # Get image summary
        summary = scene_data.get("image_summary", "")
        
        # Get metadata
        metadata = scene_data.get("metadata", {})
        
        return entities, relations, summary, metadata
    
    def _parse_triplet_new_format(self, triplet_data) -> Optional[Relation]:
        """Parse new format triplets (supports string and dict formats)"""
        import re
        
        # If dict format
        if isinstance(triplet_data, dict):
            if 'head' in triplet_data and 'predicate' in triplet_data and 'tail' in triplet_data:
                return Relation(
                    subject=triplet_data['head'],
                    predicate=triplet_data['predicate'],
                    object=triplet_data['tail']
                )
            elif 'subject' in triplet_data and 'predicate' in triplet_data and 'object' in triplet_data:
                return Relation(
                    subject=triplet_data['subject'],
                    predicate=triplet_data['predicate'],
                    object=triplet_data['object']
                )
            else:
                print(f"Warning: Unknown dict format: {triplet_data}")
                return None
        
        # If string format
        elif isinstance(triplet_data, str):
            # Match format: 【subject】-【predicate】-【object】
            pattern = r'【(.+?)】-【(.+?)】-【(.+?)】'
            match = re.match(pattern, triplet_data.strip())
            
            if match:
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                object_ = match.group(3).strip()
                return Relation(subject=subject, predicate=predicate, object=object_)
            
            # If new format doesn't match, try old format
            return self._parse_triplet_old_format(triplet_data)
        
        else:
            print(f"Warning: Unknown triplet format: {type(triplet_data)} - {triplet_data}")
            return None
    
    def _parse_triplet_old_format(self, triplet_str: str) -> Optional[Relation]:
        """Parse old format triplet string"""
        import re
        
        # Match format: subject | predicate | object
        pattern = r'(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)$'
        match = re.match(pattern, triplet_str.strip())
        
        if match:
            subject = match.group(1).strip()
            predicate = match.group(2).strip()
            object_ = match.group(3).strip()
            return Relation(subject=subject, predicate=predicate, object=object_)
        
        return None
    
    def encode_entities(self, entities: List[Entity], relations: List[Relation]) -> Tuple[np.ndarray, List[str]]:
        """Encode entity features, generate natural language descriptions and fuse structural context."""
        if not entities:
            return np.zeros((0, self.embedding_dim)), []
        
        # Build in/out edge relation map for entities
        relation_map: Dict[str, Dict[str, List[Relation]]] = {
            entity.id: {"out": [], "in": []} for entity in entities
        }
        for relation in relations:
            if relation.subject in relation_map:
                relation_map[relation.subject]["out"].append(relation)
            if relation.object in relation_map:
                relation_map[relation.object]["in"].append(relation)
        
        entity_texts: List[str] = []
        for entity in entities:
            relation_context = relation_map.get(entity.id, {"out": [], "in": []})
            description = self._build_entity_description(entity, relation_context)
            entity_texts.append(description)
        
        embeddings = self.encoder.encode(entity_texts) if entity_texts else np.zeros((0, self.embedding_dim))
        return embeddings, entity_texts
    
    def encode_relations(self, entities: List[Entity], relations: List[Relation]) -> Tuple[np.ndarray, List[str]]:
        """Encode relation features, combining subject/object semantic summaries."""
        if not relations:
            return np.zeros((0, self.embedding_dim)), []
        
        entity_lookup: Dict[str, Entity] = {entity.id: entity for entity in entities}
        relation_texts: List[str] = []
        for relation in relations:
            subject_desc = self._build_entity_brief(entity_lookup.get(relation.subject))
            object_desc = self._build_entity_brief(entity_lookup.get(relation.object))
            predicate_phrase = relation.predicate.replace("_", " ")
            relation_sentence = f"{subject_desc} {predicate_phrase} {object_desc}.".strip()
            relation_texts.append(relation_sentence)
        
        embeddings = self.encoder.encode(relation_texts) if relation_texts else np.zeros((0, self.embedding_dim))
        return embeddings, relation_texts
    
    @staticmethod
    def _format_attribute(key: str, value: str) -> str:
        """Convert attribute key-value pairs to natural language phrases."""
        key_clean = key.replace("_", " ").strip()
        if isinstance(value, str):
            value_clean = value.replace("_", " ").strip()
        else:
            value_clean = str(value)
        return f"{key_clean} is {value_clean}"
    
    def _build_entity_description(self, entity: Entity, relation_context: Dict[str, List[Relation]]) -> str:
        """Generate descriptive text based on entity and its contextual relations."""
        class_text = entity.class_name.replace("_", " ").strip() if entity.class_name else "object"
        description_parts: List[str] = [f"{entity.id} is a {class_text}."]
        
        if entity.attributes:
            attr_phrases = [
                self._format_attribute(key, value)
                for key, value in entity.attributes.items()
                if value not in (None, "", [])
            ]
            if attr_phrases:
                description_parts.append("Attributes: " + "; ".join(attr_phrases) + ".")
        
        outgoing = relation_context.get("out", []) if relation_context else []
        incoming = relation_context.get("in", []) if relation_context else []
        if outgoing:
            out_phrases = [
                f"{relation.predicate.replace('_', ' ')} {relation.object}"
                for relation in outgoing[:5]
            ]
            description_parts.append("Outgoing relations: " + "; ".join(out_phrases) + ".")
        if incoming:
            in_phrases = [
                f"{relation.subject} {relation.predicate.replace('_', ' ')}"
                for relation in incoming[:5]
            ]
            description_parts.append("Incoming relations: " + "; ".join(in_phrases) + ".")
        
        return " ".join(description_parts)
    
    def _build_entity_brief(self, entity: Optional[Entity]) -> str:
        """Construct brief statements for subject/object in relation descriptions."""
        if entity is None:
            return "an unknown entity"
        
        class_text = entity.class_name.replace("_", " ").strip() if entity.class_name else "object"
        important_attrs = []
        for key in ("type", "material", "color", "style"):
            value = entity.attributes.get(key) if entity.attributes else None
            if value:
                important_attrs.append(value.replace("_", " "))
        attr_text = ""
        if important_attrs:
            attr_text = " with " + ", ".join(important_attrs)
        
        return f"{entity.id} ({class_text}{attr_text})"
    
    def encode_summary(self, summary: str) -> np.ndarray:
        """Encode image summary"""
        if not summary:
            return np.zeros(self.embedding_dim)
        
        return self.encoder.encode([summary])[0]
    
    def build_edge_index(self, entities: List[Entity], relations: List[Relation]) -> np.ndarray:
        """Build edge index"""
        # Create mapping from entity ID to index
        entity_id_to_idx = {entity.id: i for i, entity in enumerate(entities)}
        
        edge_index = []
        for relation in relations:
            if relation.subject in entity_id_to_idx and relation.object in entity_id_to_idx:
                src_idx = entity_id_to_idx[relation.subject]
                tgt_idx = entity_id_to_idx[relation.object]
                edge_index.append([src_idx, tgt_idx])
        
        return np.array(edge_index).T if edge_index else np.array([[], []])
    
    def build_graph(self, file_id: str, scene_data: Dict) -> Dict:
        """Build a single scene graph"""
        # Parse scene data
        entities, relations, summary, metadata = self.parse_scene_data(file_id, scene_data)
        entities = self._ensure_relation_entities(entities, relations)
        
        # Encode features
        node_features, entity_texts = self.encode_entities(entities, relations)
        edge_features, relation_texts = self.encode_relations(entities, relations)
        graph_features = self.encode_summary(summary)
        
        # Build edge index
        edge_index = self.build_edge_index(entities, relations)
        
        # Create graph data
        graph_data = {
            "file_id": file_id,  # Add file ID
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "graph_features": graph_features,
            "num_nodes": len(entities),
            "num_edges": len(relations),
            "summary": summary,
            "metadata": metadata,  # Preserve metadata
            "entities": [{"id": e.id, "class": e.class_name, "attributes": e.attributes} for e in entities],
            "relations": [{"subject": r.subject, "predicate": r.predicate, "object": r.object} for r in relations],
            "entity_texts": entity_texts,
            "relation_texts": relation_texts
        }
        
        return graph_data

    def _ensure_relation_entities(self, entities: List[Entity], relations: List[Relation]) -> List[Entity]:
        """Ensure all entities referenced by relations exist; automatically add missing nodes."""
        entity_map: Dict[str, Entity] = {entity.id: entity for entity in entities}
        added: List[Entity] = []
        for relation in relations:
            for node_id in (relation.subject, relation.object):
                if not node_id:
                    continue
                if node_id not in entity_map:
                    inferred_class = self._infer_class_from_id(node_id)
                    new_entity = Entity(
                        id=node_id,
                        class_name=inferred_class,
                        attributes={"auto_generated": True}
                    )
                    entity_map[node_id] = new_entity
                    added.append(new_entity)
        if added:
            entities = entities + added
        return entities

    @staticmethod
    def _infer_class_from_id(node_id: str) -> str:
        """Simply infer class name from ID, used for auto-added nodes."""
        base = re.sub(r'_\d+$', '', node_id)
        base = base.replace('_', ' ').strip()
        return base if base else "unknown"
    
    def save_graph(self, graph_data: Dict, filename: str) -> None:
        """Save graph data"""
        save_data = {
            "file_id": graph_data["file_id"],
            "node_features": graph_data["node_features"].tolist(),
            "edge_index": graph_data["edge_index"].tolist(),
            "edge_features": graph_data["edge_features"].tolist(),
            "graph_features": graph_data["graph_features"].tolist(),
            "num_nodes": graph_data["num_nodes"],
            "num_edges": graph_data["num_edges"],
            "summary": graph_data["summary"],
            "summary_embedding": graph_data["graph_features"].tolist(),
            "metadata": graph_data["metadata"],
            "entities": graph_data["entities"],
            "relations": graph_data["relations"],
            "entity_texts": graph_data.get("entity_texts", []),
            "relation_texts": graph_data.get("relation_texts", [])
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    def process_all_scenes(self, pp2_full_data: Dict, output_dir: str, batch_size: int = 10) -> None:
        """Process all scenes - Supports GPU batch processing optimization
        
        Args:
            pp2_full_data: Scene data dictionary
            output_dir: Output directory
            batch_size: Batch size, can use larger batches on GPU
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting to process {len(pp2_full_data)} scenes...")
        
        # Use larger batch size on GPU
        if self.encoder.device == "cuda":
            batch_size = min(batch_size * 2, 50)  # GPU can handle larger batches
            print(f"✓ Using GPU batch processing, batch size: {batch_size}")
        else:
            print(f"✓ Using CPU batch processing, batch size: {batch_size}")
        
        # Use tqdm to show progress
        success_count = 0
        failed_count = 0
        
        # Process in batches
        items = list(pp2_full_data.items())
        for i in tqdm(range(0, len(items), batch_size), desc="Batch processing scene graphs"):
            batch_items = items[i:i + batch_size]
            
            # Process current batch
            for file_id, scene_data in batch_items:
                try:
                    # Build graph
                    graph_data = self.build_graph(file_id, scene_data)
                    
                    # Save graph
                    filename = output_path / f"graph_{file_id}.json"
                    self.save_graph(graph_data, str(filename))
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing scene {file_id}: {e}")
                    failed_count += 1
                    continue
            
            # GPU memory management
            if self.encoder.device == "cuda":
                torch.cuda.empty_cache()
        
        print(f"Processing complete!")
        print(f"Success: {success_count} files")
        print(f"Failed: {failed_count} files")
        print(f"Graph files saved to: {output_dir}")
        
        # Final GPU memory cleanup
        if self.encoder.device == "cuda":
            torch.cuda.empty_cache()
            print("✓ GPU memory cleared")


def main():
    """Main function - Scene graph building with GPU acceleration support"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced scene graph builder - GPU acceleration support")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cuda", "cpu"],
                       help="Computing device (auto: auto-detect, cuda: GPU, cpu: CPU)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                       help="Sentence-BERT model name")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size")
    parser.add_argument("--input", type=str, default="output/stage_02_merged/02_merged.json",
                       help="Input JSON file path")
    parser.add_argument("--output", type=str, default="output/stage_03_scene_graphs",
                       help="Output directory path")
    
    args = parser.parse_args()
    
    print("=== Enhanced Scene Graph Builder (GPU Accelerated) ===")
    print(f"Device: {args.device}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print()
    
    # Read input file
    pp2_full_file = Path(args.input)
    if not pp2_full_file.exists():
        print(f"Error: File {pp2_full_file} does not exist")
        return
    
    print(f"Reading file: {pp2_full_file}")
    with open(pp2_full_file, "r", encoding="utf-8") as f:
        pp2_full_data = json.load(f)
    
    print(f"Data contains {len(pp2_full_data)} scenes")
    
    # Create scene graph builder
    builder = EnhancedSceneGraphBuilder(
        model_name=args.model,
        device=args.device
    )
    
    # Process all scenes
    builder.process_all_scenes(
        pp2_full_data, 
        args.output,
        batch_size=args.batch_size
    )
    
    print("Enhanced scene graph building complete!")


if __name__ == "__main__":
    main()
