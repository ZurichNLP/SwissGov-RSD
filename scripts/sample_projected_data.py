#!/usr/bin/env python3
"""
Script to randomly sample a specified number of samples from projected data files.

This script reads JSONL files containing projected data (text pairs with labels)
and randomly samples a specified number of samples from them.

Usage:
    python sample_projected_data.py <input_file> [options]

Example:
    python sample_projected_data.py data/finetuning/encoder_inputs/train_de_projected.jsonl -n 50 -o sampled_data.jsonl
"""

import argparse
import json
import jsonlines
import random
import sys
from pathlib import Path


def load_jsonl_data(file_path):
    """Load data from a JSONL file."""
    data = []
    try:
        with jsonlines.open(file_path, 'r') as reader:
            for item in reader:
                data.append(item)
        print(f"Loaded {len(data)} samples from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        sys.exit(1)


def sample_data(data, n_samples, seed=None):
    """Randomly sample n_samples from the data."""
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    if len(data) < n_samples:
        print(f"Warning: Requested {n_samples} samples but only {len(data)} available. Returning all samples.")
        return data
    
    sampled = random.sample(data, n_samples)
    print(f"Sampled {len(sampled)} samples")
    return sampled


def save_jsonl_data(data, output_path):
    """Save data to a JSONL file."""
    try:
        with jsonlines.open(output_path, 'w') as writer:
            for item in data:
                writer.write(item)
        print(f"Saved {len(data)} samples to {output_path}")
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample samples from projected data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 50 samples from German projected data
  python sample_projected_data.py data/finetuning/encoder_inputs/train_de_projected.jsonl -n 50

  # Sample 100 samples with a specific seed
  python sample_projected_data.py data/finetuning/encoder_inputs/valid_fr_projected.jsonl -n 100 -s 42

  # Sample from multiple language combined file
  python sample_projected_data.py data/finetuning/encoder_inputs/train_de_it_fr_projected.jsonl -n 50 -o my_sample.jsonl
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSONL file containing projected data"
    )
    
    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        default=50,
        help="Number of samples to randomly select (default: 50)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path. If not specified, will use input filename with '_sampled' suffix"
    )
    
    parser.add_argument(
        "-s", "--seed",
        type=int,
        help="Random seed for reproducible sampling"
    )
    
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List all available projected data files and exit"
    )
    
    args = parser.parse_args()
    
    # List available files if requested
    if args.list_files:
        data_dir = Path("data/finetuning/encoder_inputs")
        if data_dir.exists():
            projected_files = list(data_dir.glob("*projected*.jsonl"))
            print("Available projected data files:")
            for file in sorted(projected_files):
                print(f"  {file}")
        else:
            print("Data directory not found: data/finetuning/encoder_inputs")
        return
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    if not input_path.suffix == '.jsonl':
        print(f"Warning: Input file '{args.input_file}' does not have .jsonl extension")
    
    # Determine output file path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_sampled{input_path.suffix}"
    
    # Load data
    data = load_jsonl_data(args.input_file)
    
    # Sample data
    sampled_data = sample_data(data, args.num_samples, args.seed)
    
    # Save sampled data
    save_jsonl_data(sampled_data, output_path)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Input file: {input_path}")
    print(f"  Output file: {output_path}")
    print(f"  Total samples in input: {len(data)}")
    print(f"  Samples selected: {len(sampled_data)}")
    if args.seed:
        print(f"  Random seed: {args.seed}")


if __name__ == "__main__":
    main()





