#!/usr/bin/env python3
"""
Script to visualize labeled text with color-coded labels.

This script takes a file with predictions (or gold labels) and a sample ID as input,
then outputs the labeled text color-coded by labels. It can also calculate Spearman
correlations with gold labels and display the raw labels.

Usage:
    python -m scripts.visualize_labeled_text <file_path> <sample_id> [--gold] [--labels] [--gold-file <gold_file>]

Examples:
    # Visualize predictions
    python -m scripts.visualize_labeled_text data/evaluation/llm_predictions/model_predictions.jsonl sample_123
    
    # Visualize gold labels
    python -m scripts.visualize_labeled_text data/evaluation/gold_labels/gold.jsonl sample_123 --gold
    
    # Visualize predictions with labels and correlation
    python -m scripts.visualize_labeled_text data/evaluation/llm_predictions/model_predictions.jsonl sample_123 --labels --gold-file data/evaluation/gold_labels/gold.jsonl
"""

import argparse
import json
import jsonlines
from pathlib import Path
from typing import List, Tuple, Optional
import sys
from scipy.stats import spearmanr

from evaluation.utils import load_predictions, load_gold_data

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def get_color_for_label(label: float) -> str:
    """
    Map label values to colors.
    
    Label ranges:
    - -1.0: Gray (unlabeled)
    - -0.9 to -0.1: Red shades (high difference)
    - 0.0 to 0.2: Yellow (low difference)
    - 0.3 to 0.5: Green (medium similarity)
    - 0.6 to 0.8: Blue (high similarity)
    - 0.9 to 1.0: Cyan (very high similarity)
    """
    if label == 0.0 <= 0.19 or label == -1.0:
        return Colors.GRAY
    elif label >= 0.2 <= 0.39:
        return Colors.YELLOW
    elif label >= 0.4 <= 0.59:
        return Colors.GREEN
    elif label >= 0.6 <= 0.79:
        return Colors.BLUE
    elif label >= 0.8 <= 0.99:
        return Colors.CYAN
    else:
        return Colors.RED

def format_labeled_text(tokens: List[str], labels: List[float], title: str = "") -> str:
    """
    Format tokens with their corresponding labels using color coding.
    """
    if not tokens or not labels:
        return f"{title}: (empty)"
    
    # Ensure lengths match
    min_len = min(len(tokens), len(labels))
    tokens = tokens[:min_len]
    labels = labels[:min_len]
    
    formatted_parts = []
    if title:
        formatted_parts.append(f"{Colors.BOLD}{title}:{Colors.RESET}")
    
    for token, label in zip(tokens, labels):
        color = get_color_for_label(label)
        formatted_parts.append(f"{color}{token}{Colors.RESET}")
    
    return " ".join(formatted_parts)

def format_labels_only(labels: List[float], title: str = "") -> str:
    """
    Format only the labels as a Python list with color coding.
    """
    if not labels:
        return f"{title}: (empty)"
    
    formatted_parts = []
    if title:
        formatted_parts.append(f"{Colors.BOLD}{title}:{Colors.RESET}")
    
    formatted_parts.append("[")
    for i, label in enumerate(labels):
        color = get_color_for_label(label)
        formatted_parts.append(f"{color}{label:.2f}{Colors.RESET}")
        if i < len(labels) - 1:
            formatted_parts.append(", ")
    formatted_parts.append("]")
    
    return "".join(formatted_parts)

def calculate_spearman_correlation(pred_labels: List[float], gold_labels: List[float]) -> Tuple[float, float]:
    """
    Calculate Spearman correlation between prediction and gold labels.
    Filters out labels where gold is -1 (consistent with evaluate_individual_correlations.py).
    """
    # Filter out labels where gold is -1
    filtered_pred = []
    filtered_gold = []
    
    for pred, gold in zip(pred_labels, gold_labels):
        if gold != -1:
            filtered_pred.append(pred)
            filtered_gold.append(gold)
    
    if len(filtered_pred) < 2:
        return None, None  # Need at least 2 points for correlation
    
    try:
        correlation, p_value = spearmanr(filtered_pred, filtered_gold)
        return correlation, p_value
    except:
        return None, None






def print_label_legend():
    """
    Print a legend explaining the color coding.
    """
    print(f"\n{Colors.BOLD}Label Color Legend:{Colors.RESET}")
    print(f"{Colors.GRAY}Gray{Colors.RESET}: Unlabeled (0.0)")
    print(f"{Colors.YELLOW}Yellow{Colors.RESET}: Low difference (0.0 to 0.2)")
    print(f"{Colors.GREEN}Green{Colors.RESET}: Medium difference (0.2 to 0.4)")
    print(f"{Colors.BLUE}Blue{Colors.RESET}: High difference (0.4 to 0.6)")
    print(f"{Colors.CYAN}Cyan{Colors.RESET}: Very high difference (0.6 to 0.8)")
    print(f"{Colors.RED}Red{Colors.RESET}: Full difference (0.8 to 1.0)")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize labeled text with color-coded labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("file_path", type=Path, help="Path to the JSONL file with predictions or gold labels")
    parser.add_argument("sample_id", type=str, help="ID of the sample to visualize")
    parser.add_argument("--gold", action="store_true", help="Treat input file as gold labels (default: predictions)")
    parser.add_argument("--no-legend", action="store_true", help="Don't print the color legend")
    parser.add_argument("--labels", action="store_true", help="Also print the color-coded labels")
    parser.add_argument("--gold-file", type=Path, help="Path to gold labels file for correlation calculation")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not args.file_path.exists():
        print(f"Error: File {args.file_path} does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Load data
    if args.gold:
        # Load gold data and find the specified sample
        gold_samples = load_gold_data(args.file_path)
        data_type = "Gold Labels"
        
        # Find sample by ID
        sample_found = False
        for gold_sample in gold_samples:
            # We need to get the ID from the original file since load_gold_data doesn't preserve IDs
            # This is a bit tricky, so let's load the original file to get IDs
            with jsonlines.open(args.file_path) as reader:
                for item in reader:
                    if item.get('id') == args.sample_id:
                        # Check if this sample matches our gold_sample
                        if (tuple(item['text_a'].split()) == gold_sample.tokens_a and 
                            tuple(item['text_b'].split()) == gold_sample.tokens_b):
                            tokens_a = list(gold_sample.tokens_a)
                            tokens_b = list(gold_sample.tokens_b)
                            labels_a = list(gold_sample.labels_a)
                            labels_b = list(gold_sample.labels_b)
                            sample_found = True
                            break
                if sample_found:
                    break
        
        if not sample_found:
            print(f"Error: Sample ID '{args.sample_id}' not found in {args.file_path}", file=sys.stderr)
            sys.exit(1)
    else:
        # Load prediction data and find the specified sample
        predictions = load_predictions(args.file_path)
        data_type = "Predictions"
        
        sample_found = False
        for prediction in predictions:
            if prediction.item_id == args.sample_id:
                diff_sample = prediction.get_difference_sample()
                tokens_a = list(diff_sample.tokens_a)
                tokens_b = list(diff_sample.tokens_b)
                labels_a = list(diff_sample.labels_a)
                labels_b = list(diff_sample.labels_b)
                sample_found = True
                break
        
        if not sample_found:
            print(f"Error: Sample ID '{args.sample_id}' not found in {args.file_path}", file=sys.stderr)
            sys.exit(1)
    
    # Print header
    print(f"{Colors.BOLD}Sample ID: {args.sample_id}{Colors.RESET}")
    print(f"{Colors.BOLD}Data Type: {data_type}{Colors.RESET}")
    print(f"{Colors.BOLD}File: {args.file_path}{Colors.RESET}")
    print("-" * 80)
    
    # Print labeled text
    text_a = format_labeled_text(tokens_a, labels_a, "Text A")
    text_b = format_labeled_text(tokens_b, labels_b, "Text B")
    
    print(text_a)
    print()
    print(text_b)
    
    # Print labels if requested
    if args.labels:
        print("\n" + "-" * 80)
        print(f"{Colors.BOLD}Labels:{Colors.RESET}")
        labels_a_formatted = format_labels_only(labels_a, "Labels A")
        labels_b_formatted = format_labels_only(labels_b, "Labels B")
        print(labels_a_formatted)
        print()
        print(labels_b_formatted)
    
    # Print statistics
    print("\n" + "-" * 80)
    print(f"{Colors.BOLD}Statistics:{Colors.RESET}")
    
    # Count labels by category
    def count_labels(labels):
        counts = {
            'unlabeled': sum(1 for l in labels if l <= 0.0),
            'low_diff': sum(1 for l in labels if 0.0 <= l <= 0.2),
            'med_diff': sum(1 for l in labels if 0.2 < l <= 0.4),
            'high_diff': sum(1 for l in labels if 0.4 < l <= 0.6),
            'very_high_diff': sum(1 for l in labels if 0.6 < l <= 0.8),
            'full_diff': sum(1 for l in labels if 0.8 < l <= 1.0)
        }
        return counts
    
    counts_a = count_labels(labels_a)
    counts_b = count_labels(labels_b)
    
    print(f"Text A ({len(tokens_a)} tokens):")
    print(f"  Unlabeled: {counts_a['unlabeled']}")
    print(f"  Low difference: {counts_a['low_diff']}")
    print(f"  Medium difference: {counts_a['med_diff']}")
    print(f"  High difference: {counts_a['high_diff']}")
    print(f"  Very high difference: {counts_a['very_high_diff']}")
    print(f"  Full difference: {counts_a['full_diff']}")
    
    print(f"\nText B ({len(tokens_b)} tokens):")
    print(f"  Unlabeled: {counts_b['unlabeled']}")
    print(f"  Low difference: {counts_b['low_diff']}")
    print(f"  Medium difference: {counts_b['med_diff']}")
    print(f"  High difference: {counts_b['high_diff']}")
    print(f"  Very high difference: {counts_b['very_high_diff']}")
    print(f"  Full difference: {counts_b['full_diff']}")
    
    # Calculate and print correlations if gold file is provided
    if args.gold_file and not args.gold:
        print("\n" + "-" * 80)
        print(f"{Colors.BOLD}Correlations with Gold Labels:{Colors.RESET}")
        
        # Load gold data using the same approach as evaluate_individual_correlations.py
        gold_samples = load_gold_data(args.gold_file)
        
        # Load the original gold data with IDs for reference
        gold_data_with_ids = []
        with open(args.gold_file, 'r') as f:
            for line in f:
                gold_data_with_ids.append(json.loads(line.strip()))
        
        # Find the sample by ID in the gold data with IDs
        gold_sample_with_id = None
        sample_index = None
        for i, item in enumerate(gold_data_with_ids):
            if item.get('id') == args.sample_id:
                gold_sample_with_id = item
                sample_index = i
                break
        
        if gold_sample_with_id is None:
            print(f"Error: Sample ID '{args.sample_id}' not found in {args.gold_file}", file=sys.stderr)
            sys.exit(1)
        
        # Get the corresponding gold sample using positional matching (same as evaluate_individual_correlations.py)
        if sample_index < len(gold_samples):
            gold_sample = gold_samples[sample_index]
            gold_labels_a = list(gold_sample.labels_a)
            gold_labels_b = list(gold_sample.labels_b)
            
            # Handle length mismatches (consistent with evaluate_individual_correlations.py)
            pred_labels_a = list(labels_a)
            pred_labels_b = list(labels_b)
            
            if len(pred_labels_a) < len(gold_labels_a):
                pred_labels_a = pred_labels_a + [0.0] * (len(gold_labels_a) - len(pred_labels_a))
            elif len(pred_labels_a) > len(gold_labels_a):
                pred_labels_a = pred_labels_a[:len(gold_labels_a)]
            
            if len(pred_labels_b) < len(gold_labels_b):
                pred_labels_b = pred_labels_b + [0.0] * (len(gold_labels_b) - len(pred_labels_b))
            elif len(pred_labels_b) > len(gold_labels_b):
                pred_labels_b = pred_labels_b[:len(gold_labels_b)]
            
            # Calculate correlation for text A
            corr_a, p_a = calculate_spearman_correlation(pred_labels_a, gold_labels_a)
            if corr_a is not None:
                print(f"Text A Spearman correlation: {corr_a:.4f} (p={p_a:.4f})")
            else:
                print("Text A: Cannot calculate correlation (insufficient data)")
            
            # Calculate correlation for text B (only if not all labels are -1)
            if not all(l == -1.0 for l in gold_labels_b):
                corr_b, p_b = calculate_spearman_correlation(pred_labels_b, gold_labels_b)
                if corr_b is not None:
                    print(f"Text B Spearman correlation: {corr_b:.4f} (p={p_b:.4f})")
                else:
                    print("Text B: Cannot calculate correlation (insufficient data)")
            else:
                print("Text B: No gold labels available for correlation")
        else:
            print(f"Gold sample at index {sample_index} not found in loaded gold data")

    # Print legend unless disabled
    if not args.no_legend:
        print_label_legend()

if __name__ == "__main__":
    main()
