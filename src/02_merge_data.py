#!/usr/bin/env python3
"""
Unified data merging script.
Supports merging JSON files and CSV data alignment functionality.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


def merge_json_files(input_dir: Path, output_file: Path, pattern: str = "*.json") -> None:
    """
    Merge JSON files in the specified directory.
    
    Args:
        input_dir: Input directory path
        output_file: Output file path
        pattern: File matching pattern, default is "*.json"
    """
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Get all matching JSON files
    json_files = list(input_dir.glob(pattern))
    
    if not json_files:
        print(f"No files matching {pattern} found in directory {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(json_files)} JSON files")
    
    merged_data: Dict[str, Any] = {}
    success_count = 0
    failed_files = []
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Use filename (without extension) as ID
                file_id = json_file.stem
                merged_data[file_id] = data
                success_count += 1
                print(f"✓ Processed: {json_file.name}")
                
        except Exception as e:
            print(f"✗ Processing failed: {json_file.name} - {e}")
            failed_files.append(json_file.name)
    
    # Save merged data
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nMerge complete!")
        print(f"Successfully processed: {success_count} files")
        print(f"Failed files: {len(failed_files)}")
        print(f"Output file: {output_file}")
        
        if failed_files:
            print(f"Failed file list: {', '.join(failed_files[:10])}{'...' if len(failed_files) > 10 else ''}")
            
    except Exception as e:
        print(f"Failed to save output file: {e}")
        sys.exit(1)


def load_csv_data(csv_file: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load CSV data, grouped by file ID.
    
    Args:
        csv_file: CSV file path
        
    Returns:
        Dictionary with file IDs as keys and related record lists as values
    """
    data = {}
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Process left_id
            left_id = row['left_id']
            if left_id not in data:
                data[left_id] = []
            
            # Add left-related record
            left_record = {
                'role': 'left',
                'opponent_id': row['right_id'],
                'winner': row['winner'],
                'lat': float(row['left_lat']),
                'long': float(row['left_long']),
                'opponent_lat': float(row['right_lat']),
                'opponent_long': float(row['right_long']),
                'category': row['category']
            }
            data[left_id].append(left_record)
            
            # Process right_id
            right_id = row['right_id']
            if right_id not in data:
                data[right_id] = []
            
            # Add right-related record
            right_record = {
                'role': 'right',
                'opponent_id': row['left_id'],
                'winner': row['winner'],
                'lat': float(row['right_lat']),
                'long': float(row['right_long']),
                'opponent_lat': float(row['left_lat']),
                'opponent_long': float(row['left_long']),
                'category': row['category']
            }
            data[right_id].append(right_record)
    
    return data


def calculate_win_rate(records: List[Dict[str, Any]], file_id: str) -> Optional[float]:
    """
    Calculate win rate.
    
    Args:
        records: List of comparison records
        file_id: File ID
        
    Returns:
        Win rate (between 0-1), or None if no valid records
    """
    wins = 0
    total = 0
    
    for record in records:
        if record['winner'] == 'equal':
            continue
        
        total += 1
        if ((record['role'] == 'left' and record['winner'] == 'left') or 
            (record['role'] == 'right' and record['winner'] == 'right')):
            wins += 1
    
    if total == 0:
        return None
    
    return wins / total


def merge_csv_json(json_file: Path, csv_files: List[Path], output_file: Path) -> None:
    """
    Merge JSON and CSV data.
    
    Args:
        json_file: JSON file path
        csv_files: List of CSV files
        output_file: Output file path
    """
    print(f"Loading JSON file: {json_file}")
    
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"JSON file contains {len(json_data)} entries")
    
    # Merge all CSV data
    all_csv_data = {}
    file_coordinates = {}  # Store coordinate information for each file
    
    for csv_file in csv_files:
        print(f"Loading CSV file: {csv_file}")
        csv_data = load_csv_data(csv_file)
        
        # Merge into total data
        for file_id, records in csv_data.items():
            if file_id not in all_csv_data:
                all_csv_data[file_id] = []
            all_csv_data[file_id].extend(records)
            
            # Extract coordinate information (take coordinates from first record)
            if file_id not in file_coordinates and records:
                first_record = records[0]
                file_coordinates[file_id] = {
                    'lat': first_record['lat'],
                    'long': first_record['long']
                }
    
    print(f"CSV data contains {len(all_csv_data)} unique file IDs")
    
    # Merge data
    merged_data = {}
    matched_count = 0
    unmatched_json = 0
    unmatched_csv = 0
    
    # Process each entry in JSON
    for file_id, json_entry in json_data.items():
        merged_entry = json_entry.copy()
        
        # Add CSV data (if exists)
        if file_id in all_csv_data:
            # Get coordinate information
            coordinates = file_coordinates.get(file_id, {'lat': None, 'long': None})
            
            merged_entry['metadata'] = {
                'coordinates': coordinates,
                'comparisons': all_csv_data[file_id],
                'total_comparisons': len(all_csv_data[file_id]),
                'categories': list(set(record['category'] for record in all_csv_data[file_id])),
                'win_rate': calculate_win_rate(all_csv_data[file_id], file_id)
            }
            matched_count += 1
        else:
            merged_entry['metadata'] = {
                'coordinates': {'lat': None, 'long': None},
                'comparisons': [],
                'total_comparisons': 0,
                'categories': [],
                'win_rate': None
            }
            unmatched_json += 1
        
        merged_data[file_id] = merged_entry
    
    # Count unmatched CSV entries
    for file_id in all_csv_data:
        if file_id not in json_data:
            unmatched_csv += 1
    
    # Save merged data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nMerge complete!")
    print(f"Matched files: {matched_count}")
    print(f"Unmatched files in JSON: {unmatched_json}")
    print(f"Unmatched files in CSV: {unmatched_csv}")
    print(f"Output file: {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified data merging tool - Supports JSON file merging and CSV data alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Merge JSON files
  python merge_data.py json output/ -o merged.json
  
  # CSV data alignment
  python merge_data.py csv pp2.json data.csv -o pp2_full.json
  
  # CSV data alignment (multiple CSV files)
  python merge_data.py csv pp2.json data1.csv data2.csv -o pp2_full.json
        """
    )
    
    # Common arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Merge mode')
    
    # JSON merge mode
    json_parser = subparsers.add_parser('json', help='Merge JSON files')
    json_parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing JSON files"
    )
    json_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("merged_output.json"),
        help="Output file path (default: merged_output.json)"
    )
    json_parser.add_argument(
        "-p", "--pattern",
        type=str,
        default="*.json",
        help="File matching pattern (default: *.json)"
    )
    
    # CSV alignment mode
    csv_parser = subparsers.add_parser('csv', help='CSV data alignment')
    csv_parser.add_argument(
        "json_file",
        type=Path,
        help="JSON file path"
    )
    csv_parser.add_argument(
        "csv_files",
        nargs='+',
        type=Path,
        help="List of CSV file paths"
    )
    csv_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("merged_output.json"),
        help="Output file path (default: merged_output.json)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    if not args.mode:
        print("Error: Please specify merge mode (json or csv)")
        print("Use --help to see detailed help")
        sys.exit(1)
    
    if args.verbose:
        print(f"Mode: {args.mode}")
        if args.mode == 'json':
            print(f"Input directory: {args.input_dir}")
            print(f"Output file: {args.output}")
            print(f"Match pattern: {args.pattern}")
        else:
            print(f"JSON file: {args.json_file}")
            print(f"CSV files: {args.csv_files}")
            print(f"Output file: {args.output}")
        print()
    
    try:
        if args.mode == 'json':
            merge_json_files(args.input_dir, args.output, args.pattern)
        elif args.mode == 'csv':
            merge_csv_json(args.json_file, args.csv_files, args.output)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
