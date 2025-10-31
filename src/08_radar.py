#!/usr/bin/env python3
"""
Radar chart visualization script for comparing performance metrics of different models.

Supports input of multiple model names and corresponding metric values, generates radar charts for comparison.
Uses Nature journal style color scheme.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Nature style color scheme (consistent with 07_evaluate_and_visualize.py)
NATURE_PALETTE = ['#1f4e5f', '#326273', '#488a99', '#6ba292', '#a3c9a8']
ACCENT_COLOR = '#d1495b'
BACKGROUND_COLOR = '#f4f4f2'
PERCEPTION_DIMENSIONS = ["safety", "lively", "boring", "wealthy", "depressing", "beautiful"]
PERFORMANCE_METRICS = ["accuracy", "precision", "auc", "f1", "recall"]


def apply_nature_style() -> None:
    """Configure matplotlib/seaborn for Nature/Science publication style."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#2a2a2a',
        'axes.labelcolor': '#1a1a1a',
        'text.color': '#1a1a1a',
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'axes.labelsize': 16,
        'xtick.color': '#1a1a1a',
        'ytick.color': '#1a1a1a',
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.weight': 'bold',
        'grid.color': '#d0d0d0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fontsize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'lines.linewidth': 3.0,
        'patch.linewidth': 0.0,
    })
    sns.set_style('whitegrid')


def order_metrics(metrics: List[str]) -> List[str]:
    preferred = [m for m in PERCEPTION_DIMENSIONS if m in metrics]
    remaining = [m for m in metrics if m not in preferred]
    return preferred + remaining


def create_radar_chart(
    models_data: Dict[str, Dict[str, float]],
    metrics: List[str],
    output_path: Path,
    title: str = "Model Performance Comparison Radar Chart",
    show_value_labels: bool = False,
    axis_labels: Optional[List[str]] = None
) -> None:
    """
    Create radar chart comparing performance metrics of multiple models.
    
    Args:
        models_data: Dictionary of model data in format {model_name: {metric: value}}
        metrics: List of metrics to display
        output_path: Output file path
        title: Chart title
        show_value_labels: Whether to annotate each vertex with its numeric value
        axis_labels: Optional tick labels for the metric axes
    """
    # Set angles for polar plot
    base_angles = [n / len(metrics) * 2 * math.pi for n in range(len(metrics))]
    angles = base_angles + base_angles[:1]  # Close the polygon
    labels = axis_labels if axis_labels and len(axis_labels) == len(metrics) else metrics
    
    # Create figure with larger size for publication quality
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Enhanced color palette for scientific publications
    colors = [
        '#2578B5',  # diagram blue
        '#D84B39',  # diagram red
        '#F39C1F',  # diagram orange
        '#4AA54A',  # diagram green
        '#9A62C7',  # diagram violet
        '#5AB1C9',  # diagram cyan
        '#E48CC7',  # diagram pink
        '#A5BE5B',  # diagram lime
    ]
    specific_colors = {
        'graphmae': '#D84B39',
        'cnn': '#9A62C7',
    }
    
    # Plot each model
    model_count = len(models_data)

    for i, (model_name, model_metrics) in enumerate(models_data.items()):
        values = []
        for metric in metrics:
            raw_value = model_metrics.get(metric, 0.0)
            if raw_value is None:
                raw_value = 0.0
            try:
                values.append(float(raw_value))
            except (TypeError, ValueError):
                values.append(0.0)
        closed_values = values + values[:1]
        
        # Plot lines with enhanced styling
        color_key = model_name.lower()
        color = specific_colors.get(color_key, colors[i % len(colors)])
        line_width = 3.5 if color_key == 'graphmae' else 2.25
        ax.plot(angles, closed_values, '-', linewidth=line_width, label=model_name,
                color=color)
        
        # Fill area with transparency
        ax.fill(angles, closed_values, alpha=0.18, color=color)

        if show_value_labels:
            radial_offset = (i - (model_count - 1) / 2) * 0.03
            for angle, val in zip(base_angles, values):
                label_radius = min(max(val + radial_offset, 0.02), 1.0)
                ax.text(
                    angle,
                    label_radius,
                    f"{val:.2f}",
                    color=color,
                    fontsize=10,
                    fontweight='bold',
                    ha='center',
                    va='center'
                )
    
    # Set labels with enhanced formatting
    ax.set_xticks(base_angles)
    ax.set_xticklabels(labels, fontweight='bold')
    
    # Set y-axis range and ticks
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    
    # Enhanced grid styling
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Add title with enhanced styling
    plt.title(title, size=20, fontweight='bold', pad=30)
    
    # Enhanced legend
    ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05),
              fontsize=13, frameon=True, fancybox=False, shadow=False,
              borderpad=0.6, prop={'weight': 'bold'})
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high resolution for publication
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Radar chart saved to: {output_path}")


def parse_model_input() -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Parse user input for model data.
    
    Returns:
        Dictionary of model data and list of metrics
    """
    print("Please enter model data (press Enter twice to finish):")
    print("Format: ModelName,metric1:value1,metric2:value2,...")
    print("Example: MyModel,accuracy:0.85,auc:0.92,f1:0.88")
    print()
    
    models_data = {}
    all_metrics = set()
    
    while True:
        try:
            line = input().strip()
            if not line:
                break
            
            parts = line.split(',')
            if len(parts) < 2:
                print("Format error, please re-enter")
                continue
            
            model_name = parts[0].strip()
            metrics_dict = {}
            
            for part in parts[1:]:
                if ':' not in part:
                    print(f"Metric format error: {part}")
                    continue
                
                metric, value = part.split(':', 1)
                metric = metric.strip()
                try:
                    value = float(value.strip())
                    metrics_dict[metric] = value
                    all_metrics.add(metric)
                except ValueError:
                    print(f"Value format error: {value}")
                    continue
            
            if metrics_dict:
                models_data[model_name] = metrics_dict
                print(f"Added model: {model_name}")
            
        except KeyboardInterrupt:
            print("\nInput interrupted")
            break
        except EOFError:
            break
    
    metrics_list = sorted(list(all_metrics))
    return models_data, metrics_list


def load_from_json(json_path: str) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Load model data from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary of model data and list of metrics
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models_data = data.get('models', {})
    metrics = data.get('metrics', [])
    
    return models_data, metrics


def load_performance_from_files(
    model_names: Iterable[str],
    metrics_dir: Path,
    metric_keys: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Load performance metrics from individual metrics_{model}.json files.

    Args:
        model_names: Iterable of model identifiers to load.
        metrics_dir: Directory containing metrics files.
        metric_keys: Ordered list of metric keys to extract.

    Returns:
        Dictionary of model performance metrics.
    """
    aggregated: Dict[str, Dict[str, float]] = {}

    for model_name in model_names:
        file_path = metrics_dir / f"metrics_{model_name.lower()}.json"
        if not file_path.exists():
            print(f"Metrics file missing for model '{model_name}': {file_path}")
            continue

        try:
            raw_data = json.loads(file_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            print(f"Failed to parse metrics file for '{model_name}': {exc}")
            continue

        model_metrics: Dict[str, float] = {}
        for key in metric_keys:
            value = raw_data.get(key)
            try:
                model_metrics[key] = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                model_metrics[key] = 0.0

        if model_metrics:
            aggregated[model_name] = model_metrics

    return aggregated


def create_sample_data() -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Create sample data for testing.
    
    Returns:
        Sample model data dictionary and metrics list
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap']
    
    models_data = {
        'Proposed Model': {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1': 0.85,
            'auc': 0.92,
            'ap': 0.89
        },
        'Baseline': {
            'accuracy': 0.75,
            'precision': 0.73,
            'recall': 0.77,
            'f1': 0.75,
            'auc': 0.81,
            'ap': 0.78
        },
        'Random': {
            'accuracy': 0.50,
            'precision': 0.50,
            'recall': 0.50,
            'f1': 0.50,
            'auc': 0.50,
            'ap': 0.50
        }
    }
    
    return models_data, metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate model performance comparison radar chart')
    parser.add_argument('--input-file', type=str, default='result/model_accuracy_summary.json',
                       help='JSON input file path (default: result/model_accuracy_summary.json)')
    parser.add_argument('--output', type=str, default='result/radar_comparison.png', 
                       help='Output image path')
    parser.add_argument('--title', type=str, default='Model Performance Comparison Radar Chart', 
                       help='Chart title')
    parser.add_argument('--performance-title', type=str, default='Classification Metrics Radar Chart',
                       help='Title for the additional classification metrics radar chart')
    parser.add_argument('--performance-output', type=str, default='result/radar_classification_metrics.png',
                       help='Output image path for classification metrics radar chart')
    parser.add_argument('--metrics-dir', type=str, default='result',
                       help='Directory containing metrics_*.json files')
    parser.add_argument('--skip-performance', action='store_true',
                       help='Skip generating the classification metrics radar chart')
    parser.add_argument('--hide-value-labels', action='store_true',
                        help='Do not annotate vertex values on the radar charts')
    parser.add_argument('--interactive', action='store_true', 
                       help='Interactive model data input')
    parser.add_argument('--sample', action='store_true', 
                       help='Use sample data')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Apply Nature style
    apply_nature_style()
    
    # Get model data
    input_path = Path(args.input_file) if args.input_file else None
    if args.sample:
        models_data, metrics = create_sample_data()
        print("Using sample data")
    elif input_path and input_path.exists():
        models_data, metrics = load_from_json(str(input_path))
        print(f"Loading data from file: {input_path}")
    elif args.interactive:
        models_data, metrics = parse_model_input()
    else:
        if input_path:
            print(f"Input file not found: {input_path}")
        print("Please prepare summary JSON via 06_comparison_trainer.py or use --interactive / --sample.")
        return
    
    if not models_data:
        print("No model data found, exiting")
        return
    
    if not metrics:
        metrics = sorted({metric for model_metrics in models_data.values() for metric in model_metrics.keys()})
    metrics = order_metrics(metrics)
    if not metrics:
        print("No metrics available for radar chart.")
        return
    
    print(f"Found {len(models_data)} models")
    print(f"Metrics: {', '.join(metrics)}")
    
    # Create radar chart
    output_path = Path(args.output)
    create_radar_chart(
        models_data,
        metrics,
        output_path,
        args.title,
        show_value_labels=not args.hide_value_labels,
        axis_labels=[''] * len(metrics)
    )
    
    # Save data to JSON file
    data_file = output_path.with_suffix('.json')
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump({
            'models': models_data,
            'metrics': metrics,
            'title': args.title
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Data saved to: {data_file}")

    if args.skip_performance:
        return

    performance_metrics = PERFORMANCE_METRICS.copy()
    metrics_dir = Path(args.metrics_dir)
    performance_data = load_performance_from_files(models_data.keys(), metrics_dir, performance_metrics)

    if not performance_data:
        print("No classification metrics data found; skipping additional radar chart.")
        return

    performance_output = Path(args.performance_output)
    shorthand_map = {
        'accuracy': 'acc',
        'precision': 'precision',
        'auc': 'auc',
        'f1': 'f1',
        'recall': 'recall',
    }
    performance_axis_labels = [shorthand_map.get(metric, metric) for metric in performance_metrics]

    create_radar_chart(
        performance_data,
        performance_metrics,
        performance_output,
        args.performance_title,
        show_value_labels=not args.hide_value_labels,
        axis_labels=performance_axis_labels
    )

    performance_data_file = performance_output.with_suffix('.json')
    with open(performance_data_file, 'w', encoding='utf-8') as f:
        json.dump({
            'models': performance_data,
            'metrics': performance_metrics,
            'title': args.performance_title
        }, f, ensure_ascii=False, indent=2)

    print(f"Classification metrics radar chart saved to: {performance_output}")
    print(f"Classification data saved to: {performance_data_file}")


if __name__ == '__main__':
    main()
