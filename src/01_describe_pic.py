#!/usr/bin/env python3
"""
Test Gemini 2.5 Flash API connection and functionality.
Generate scene descriptions for images using vision-language models.
"""
import json
import base64
import csv
import os
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from queue import Queue, Empty
from threading import Thread, Lock

API_BASE = os.getenv("API_BASE", "https://chatapi.nloli.xyz")
MODEL = os.getenv("MODEL", "gemini-2.5-flash")

# Read API keys from environment variables, supporting multiple keys separated by commas
def get_api_keys() -> List[str]:
    """Read API keys from environment variables."""
    api_keys_str = os.getenv("API_KEYS", "")
    if not api_keys_str:
        print("Warning: API_KEYS environment variable not set. Please configure it before use.")
        return []
    # Support multiple keys separated by commas
    keys = [key.strip() for key in api_keys_str.split(",") if key.strip()]
    return keys

API_KEYS: List[str] = get_api_keys()
PER_KEY_WORKERS = int(os.getenv("PER_KEY_WORKERS", "1"))
IMAGE_DIR = Path("data/PP2/final_photo_dataset")
METADATA_PATH = Path("data/PP2/metadata/final_data.csv")
OUTPUT_DIR = Path("output/stage_01_descriptions/PP2")
LOG_PATH = Path("logs/describe_pic_errors.log")

def mask_key(key: str) -> str:
    """Return a masked version of the key for display purposes."""
    if len(key) <= 10:
        return key
    return f"{key[:6]}...{key[-4:]}"


def find_image_path(image_id: str, image_dir: Path) -> Optional[Path]:
    """Find image file in directory based on ID."""
    candidates = [
        image_dir / f"{image_id}{ext}"
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fallback: if filename already contains extension
    direct_path = image_dir / image_id
    if direct_path.exists():
        return direct_path
    return None


def log_error(message: str) -> None:
    """Log error message to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")


def test_api_connection(api_key: str, api_base: str = API_BASE, model: str = MODEL) -> bool:
    """Test API connection."""
    
    # Build endpoint
    endpoint = f"{api_base}/v1beta/models/{model}:generateContent"
    
    # Test request
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Hello, please respond with 'API connection successful'"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 100,
            "responseMimeType": "application/json",
        }
    }
    
    try:
        print(f"Testing API connection... (key: {mask_key(api_key)})")
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model}")
        
        response = requests.post(
            endpoint, 
            headers=headers, 
            params=params, 
            json=payload, 
            timeout=30,
            verify=False
        )
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ API connection successful!")
            print(f"Response content: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"✗ API connection failed: {response.status_code}")
            print(f"Error message: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Connection exception: {e}")
        return False


def process_pp2_images(
    max_images: int = 8000,
    api_keys: Optional[List[str]] = None,
    per_key_workers: int = PER_KEY_WORKERS,
    api_base: str = API_BASE,
    model: str = MODEL,
) -> bool:
    """Process PP2 images in parallel based on pairing metadata."""
    if api_keys is None:
        api_keys = API_KEYS
    if not api_keys:
        print("✗ No API Key provided, unable to process images")
        return False
    if per_key_workers < 1:
        print("✗ Number of parallel workers per key must be greater than 0")
        return False
    
    if not METADATA_PATH.exists():
        print(f"✗ Metadata file does not exist: {METADATA_PATH}")
        return False
    if not IMAGE_DIR.exists():
        print(f"✗ Image directory does not exist: {IMAGE_DIR}")
        return False
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    seen_ids = set()
    tasks: List[Path] = []
    skipped_existing = 0
    missing_files = 0
    duplicate_ids = 0
    total_pairs = 0
    
    with METADATA_PATH.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None or "left_id" not in reader.fieldnames or "right_id" not in reader.fieldnames:
            print("✗ Metadata file missing left_id/right_id fields")
            return False
        
        for row in reader:
            total_pairs += 1
            for side in ("left_id", "right_id"):
                image_id = (row.get(side) or "").strip()
                if not image_id:
                    continue
                if image_id in seen_ids:
                    duplicate_ids += 1
                    continue
                seen_ids.add(image_id)
                
                image_path = find_image_path(image_id, IMAGE_DIR)
                if not image_path:
                    missing_files += 1
                    print(f"⚠ Image file not found: {image_id}")
                    continue
                
                output_file = OUTPUT_DIR / f"{image_path.stem}.json"
                if output_file.exists():
                    skipped_existing += 1
                    continue
                
                tasks.append(image_path)
                if max_images > 0 and len(tasks) >= max_images:
                    break
            if max_images > 0 and len(tasks) >= max_images:
                break
    
    if not tasks:
        print("✗ No images found to process (all may already have JSON files or files are missing)")
        return False
    
    print(
        f"Read {total_pairs} pairing records, filtered to {len(tasks)} images to process "
        f"(existing JSON: {skipped_existing}, missing files: {missing_files}, duplicate IDs: {duplicate_ids})"
    )
    
    task_queue: "Queue[Path]" = Queue()
    for image_path in tasks:
        task_queue.put(image_path)
    
    success_count = 0
    failed_count = 0
    results_lock = Lock()
    total_targets = len(tasks)
    progress_lock = Lock()
    progress_state = {
        "processed": 0,
        "success": 0,
        "failed": 0,
        "skipped_existing": skipped_existing,
        "skipped_runtime": 0,
        "total": total_targets,
    }

    def render_progress(state: dict) -> str:
        return (
            f"[Progress] {state['processed']}/{state['total']} processed | "
            f"success: {state['success']} | failed: {state['failed']} | "
            f"skipped(existing): {state['skipped_existing']} | "
            f"skipped(runtime): {state['skipped_runtime']}"
        )

    def update_progress(delta_processed=0, delta_success=0, delta_failed=0, delta_skipped_runtime=0) -> None:
        with progress_lock:
            progress_state["processed"] += delta_processed
            progress_state["success"] += delta_success
            progress_state["failed"] += delta_failed
            progress_state["skipped_runtime"] += delta_skipped_runtime
            processed = progress_state["processed"]
            total = progress_state["total"]
            if (
                delta_processed
                and (processed == total or processed % 10 == 0)
            ) or delta_skipped_runtime:
                print(render_progress(progress_state))

    print(render_progress(progress_state))
    
    def worker(api_key: str, worker_name: str) -> None:
        nonlocal success_count, failed_count
        while True:
            try:
                image_path: Path = task_queue.get(timeout=1)
            except Empty:
                break
            
            try:
                print(f"\n[{worker_name}] Starting processing: {image_path.name}")
                output_file = OUTPUT_DIR / f"{image_path.stem}.json"
                if output_file.exists():
                    print(f"[{worker_name}] JSON already exists, skipping: {image_path.name}")
                    update_progress(delta_processed=1, delta_skipped_runtime=1)
                    continue
                success = process_single_image(image_path, api_key, api_base, model)
                with results_lock:
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                if success:
                    update_progress(delta_processed=1, delta_success=1)
                else:
                    update_progress(delta_processed=1, delta_failed=1)
            except Exception as exc:
                with results_lock:
                    failed_count += 1
                update_progress(delta_processed=1, delta_failed=1)
                print(f"[{worker_name}] ✗ Processing exception {image_path.name}: {exc}")
                log_error(f"[{worker_name}] Exception processing {image_path.name}: {exc}")
            finally:
                task_queue.task_done()
    
    threads: List[Thread] = []
    for key_index, api_key in enumerate(api_keys, start=1):
        for worker_index in range(1, per_key_workers + 1):
            worker_name = f"Key{key_index}-T{worker_index}"
            thread = Thread(target=worker, args=(api_key, worker_name), daemon=True)
            thread.start()
            threads.append(thread)
    
    task_queue.join()
    
    for thread in threads:
        thread.join()
    
    print(render_progress(progress_state))
    
    print(
        f"\nProcessing complete: {success_count} succeeded, {failed_count} failed, "
        f"total {success_count + failed_count} / {total_targets} images"
    )
    print(
        f"Skipped (existing JSON): {progress_state['skipped_existing']} images, "
        f"skipped at runtime: {progress_state['skipped_runtime']} images"
    )
    return success_count > 0


def process_single_image(image_path: Path, api_key: str, api_base: str, model: str) -> bool:
    """Process a single image."""
    
    # Encode image
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        print(f"Image encoding complete, size: {len(image_base64)} characters")
    except Exception as e:
        print(f"✗ Image encoding failed: {e}")
        return False
    
    # Build request
    endpoint = f"{api_base}/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": """---
You are an AI expert specializing in semantic scene analysis for autonomous driving systems.

Task
Analyze the provided street-view image (or its structured textual description) and output a single JSON object describing the scene as semantic triplets. You must follow the procedure below and satisfy all hard constraints.

Hard constraints

Connectivity: No isolated entities; every entity must appear in ≥1 relation. If an entity cannot form a credible relation, omit it.

De-duplication & hierarchy: One entity ID per physical instance. Put brand/model/color and other fine-grained tags into that entity’s attributes; never create alias entities. For groups, list members only if they have independent relations/actions and connect them via member_of.

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

Use “stuff” anchors (road/sidewalk/facade/wall) to help connectivity.

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

If counts are given coarsely (e.g., “many”), create groups with count/density and relate members only when they have distinct actions.

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
"""},
                    {
                        "text": (
                            "Follow every constraint above and respond with a single JSON object that includes "
                            "exactly the keys image_summary, entities, and semantic_triplets. Do not include any "
                            "extra commentary or formatting."
                        )
                    },
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
        print("Testing image analysis...")
        
        response = requests.post(
            endpoint,
            headers=headers,
            params=params,
            json=payload,
            timeout=60,
            verify=False
        )
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Image analysis successful!")
            
            # Extract text content
            candidates = result.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                finish_reason = candidates[0].get("finishReason")
                for part in parts:
                    if "text" in part:
                        print(f"Analysis result:")
                        print(part['text'])
                        print("=" * 50)
                        
                        # Try to parse JSON
                        try:
                            # Handle possible Markdown code block format
                            text_content = part['text'].strip()
                            if text_content.startswith('```json'):
                                # Remove ```json and ``` markers
                                text_content = text_content[7:]  # Remove ```json
                                if text_content.endswith('```'):
                                    text_content = text_content[:-3]  # Remove ```
                                text_content = text_content.strip()
                            elif text_content.startswith('```'):
                                # Remove ``` markers
                                text_content = text_content[3:]
                                if text_content.endswith('```'):
                                    text_content = text_content[:-3]
                                text_content = text_content.strip()
                            
                            json_data = json.loads(text_content)
                            print("✓ JSON parsing successful!")
                            print("Parsed JSON:")
                            print(json.dumps(json_data, indent=2, ensure_ascii=False))
                            
                            # Save to file
                            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                            
                            # Use original image filename (without extension) as JSON filename
                            output_file = OUTPUT_DIR / f"{image_path.stem}.json"
                            
                            with open(output_file, "w", encoding="utf-8") as f:
                                json.dump(json_data, f, ensure_ascii=False, indent=2)
                            
                            print(f"✓ Results saved to: {output_file}")
                            return True
                        except json.JSONDecodeError as e:
                            print(f"✗ JSON parsing failed: {e}")
                            print("Processed content first 200 characters:")
                            print(repr(text_content[:200]))
                            if len(text_content) > 200:
                                print("...")
                            log_error(
                                f"{image_path.name}: JSON decode error ({e}); finish_reason={finish_reason}; "
                                f"text_length={len(text_content)}; snippet={repr(text_content[:200])}"
                            )
                            break
                else:
                    log_error(f"{image_path.name}: No parseable text part found in response, finish_reason={finish_reason}")
                    return False
                return False
            else:
                log_error(f"{image_path.name}: Response missing candidates field, original keys: {list(result.keys())}")
                return False
        else:
            print(f"✗ Image analysis failed: {response.status_code}")
            print(f"Error message: {response.text}")
            log_error(
                f"{image_path.name}: HTTP {response.status_code} error, response content: {response.text[:500]!r}"
            )
            return False
            
    except Exception as e:
        print(f"✗ Image analysis exception: {e}")
        log_error(f"{image_path.name}: Processing exception {e}")
        return False


def main():
    """Main test function."""
    print("=== Gemini 2.5 Flash API Test ===\n")
    
    # Test all API Keys
    if not API_KEYS:
        print("✗ No API Keys configured, cannot start test")
        return
    
    print("1. Testing API connection")
    available_keys: List[str] = []
    for idx, key in enumerate(API_KEYS, start=1):
        print(f"\n- Testing Key {idx}/{len(API_KEYS)} ({mask_key(key)})")
        if test_api_connection(key):
            available_keys.append(key)
        else:
            print(f"✗ Key {mask_key(key)} connection failed")
    
    if not available_keys:
        print("\n✗ All API Keys failed to connect, task terminated")
        return
    
    # Process images in parallel
    print(f"\n2. Processing PP2 images in parallel (workers per key: "
          f"{PER_KEY_WORKERS}, number of keys used: {len(available_keys)})")
    pp2_ok = process_pp2_images(
        max_images=8000,
        api_keys=available_keys,
        per_key_workers=PER_KEY_WORKERS,
    )
    print()
    
    if pp2_ok:
        print("✓ PP2 image processing complete!")
    else:
        print("✗ PP2 image processing encountered issues (please check logs)")


if __name__ == "__main__":
    main()
