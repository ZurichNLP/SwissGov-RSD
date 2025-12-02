#!/usr/bin/env python3
"""
Test script to validate all splits created by split_admin.py

This script checks:
1. Test sets contain the same data across all file types (using document IDs)
2. Dev sets contain the same data across all file types (using document IDs)
3. Dev and test sets don't overlap
4. Train sets contain the same data across all file types
5. Validation sets contain the same data across all file types
6. Train and validation sets don't overlap
"""

import jsonlines
import glob
import os
import argparse
import re
from collections import defaultdict

def extract_sample_id(document_id):
    """Extract the numeric sample ID from a document ID (e.g., 'admin_de_1' -> '1')."""
    match = re.search(r'admin_[a-z]{2}_(\d+)$', document_id)
    if match:
        return int(match.group(1))
    return None

def load_sample_ids(file_path):
    """Load all numeric sample IDs from a jsonl file."""
    sample_ids = set()
    try:
        with jsonlines.open(file_path) as f:
            for line in f:
                if "id" in line:
                    sample_id = extract_sample_id(line["id"])
                    if sample_id is not None:
                        sample_ids.add(sample_id)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()
    return sample_ids

def get_file_type_and_lang(filename):
    """Extract file type and language from filename."""
    if "short" in filename:
        if "gold_admin" in filename:
            match = re.search(r"gold_admin_([a-z]{2})_short", filename)
            if match:
                return "short_admin", match.group(1)
        else:
            match = re.search(r"admin_([a-z]{2})_short", filename)
            if match:
                return "short_admin", match.group(1)
    elif "test_admin" in filename:
        match = re.search(r"test_admin_([a-z]{2})", filename)
        if match:
            return "llm_input", match.group(1)
    elif "admin" in filename:
        match = re.search(r"admin_([a-z]{2})", filename)
        if match:
            return "admin", match.group(1)
    
    return "unknown", "unknown"

def test_splits(base_dir, test_llm_inputs=False, test_short=False):
    """Test all splits for consistency and correctness."""
    
    print(f"Testing splits in: {base_dir}")
    if test_llm_inputs:
        print("Mode: LLM inputs")
    elif test_short:
        print("Mode: Short admin files")
    else:
        print("Mode: Regular admin files")
    print("-" * 50)
    
    # Determine which directories to test
    if test_llm_inputs:
        test_dir = os.path.join(base_dir, "llm_inputs", "test")
        dev_dir = os.path.join(base_dir, "llm_inputs", "dev")
        train_dir = os.path.join(base_dir, "llm_inputs", "dev", "train")
        val_dir = os.path.join(base_dir, "llm_inputs", "dev", "val")
    elif test_short:
        test_dir = os.path.join(base_dir, "gold_labels", "test")
        dev_dir = os.path.join(base_dir, "gold_labels", "dev")
        train_dir = os.path.join(base_dir, "gold_labels", "dev", "train")
        val_dir = os.path.join(base_dir, "gold_labels", "dev", "val")
    else:
        test_dir = os.path.join(base_dir, "gold_labels", "test")
        dev_dir = os.path.join(base_dir, "gold_labels", "dev")
        train_dir = os.path.join(base_dir, "gold_labels", "dev", "train")
        val_dir = os.path.join(base_dir, "gold_labels", "dev", "val")
    
    # Test 1: Test sets contain the same data across all languages
    print("\n1. Testing test sets consistency...")
    test_files = glob.glob(os.path.join(test_dir, "*.jsonl"))
    test_sample_ids_by_lang = {}
    
    for test_file in test_files:
        filename = os.path.basename(test_file)
        file_type, lang = get_file_type_and_lang(filename)
        if lang != "unknown":
            sample_ids = load_sample_ids(test_file)
            test_sample_ids_by_lang[lang] = sample_ids
            print(f"  {lang}: {len(sample_ids)} sample IDs")
    
    # Check if all test sets have the same sample IDs
    if len(test_sample_ids_by_lang) > 1:
        first_lang = list(test_sample_ids_by_lang.keys())[0]
        first_sample_ids = test_sample_ids_by_lang[first_lang]
        for lang, sample_ids in test_sample_ids_by_lang.items():
            if sample_ids != first_sample_ids:
                print(f"  ❌ Test set mismatch: {lang} differs from {first_lang}")
                missing = first_sample_ids - sample_ids
                extra = sample_ids - first_sample_ids
                if missing:
                    print(f"    Missing in {lang}: {len(missing)} sample IDs")
                if extra:
                    print(f"    Extra in {lang}: {len(extra)} sample IDs")
            else:
                print(f"  ✅ Test set {lang} matches {first_lang}")
    
    # Test 2: Dev sets contain the same data across all languages
    print("\n2. Testing dev sets consistency...")
    dev_files = glob.glob(os.path.join(dev_dir, "*.jsonl"))
    dev_sample_ids_by_lang = {}
    
    for dev_file in dev_files:
        filename = os.path.basename(dev_file)
        file_type, lang = get_file_type_and_lang(filename)
        if lang != "unknown":
            sample_ids = load_sample_ids(dev_file)
            dev_sample_ids_by_lang[lang] = sample_ids
            print(f"  {lang}: {len(sample_ids)} sample IDs")
    
    # Check if all dev sets have the same sample IDs
    if len(dev_sample_ids_by_lang) > 1:
        first_lang = list(dev_sample_ids_by_lang.keys())[0]
        first_sample_ids = dev_sample_ids_by_lang[first_lang]
        for lang, sample_ids in dev_sample_ids_by_lang.items():
            if sample_ids != first_sample_ids:
                print(f"  ❌ Dev set mismatch: {lang} differs from {first_lang}")
                missing = first_sample_ids - sample_ids
                extra = sample_ids - first_sample_ids
                if missing:
                    print(f"    Missing in {lang}: {len(missing)} sample IDs")
                if extra:
                    print(f"    Extra in {lang}: {len(extra)} sample IDs")
            else:
                print(f"  ✅ Dev set {lang} matches {first_lang}")
    
    # Test 3: Dev and test sets don't overlap
    print("\n3. Testing dev-test overlap...")
    if dev_sample_ids_by_lang and test_sample_ids_by_lang:
        first_dev_lang = list(dev_sample_ids_by_lang.keys())[0]
        first_test_lang = list(test_sample_ids_by_lang.keys())[0]
        dev_sample_ids = dev_sample_ids_by_lang[first_dev_lang]
        test_sample_ids = test_sample_ids_by_lang[first_test_lang]
        
        overlap = dev_sample_ids.intersection(test_sample_ids)
        if overlap:
            print(f"  ❌ Dev and test sets overlap: {len(overlap)} common sample IDs")
            print(f"    Overlapping sample IDs: {sorted(list(overlap))[:10]}...")
        else:
            print(f"  ✅ Dev and test sets are disjoint (no overlap)")
    
    # Test 4: Train sets contain the same data across all languages
    print("\n4. Testing train sets consistency...")
    if os.path.exists(train_dir):
        train_files = glob.glob(os.path.join(train_dir, "*.jsonl"))
        train_sample_ids_by_lang = {}
        
        for train_file in train_files:
            filename = os.path.basename(train_file)
            file_type, lang = get_file_type_and_lang(filename)
            if lang != "unknown":
                sample_ids = load_sample_ids(train_file)
                train_sample_ids_by_lang[lang] = sample_ids
                print(f"  {lang}: {len(sample_ids)} sample IDs")
        
        # Check if all train sets have the same sample IDs
        if len(train_sample_ids_by_lang) > 1:
            first_lang = list(train_sample_ids_by_lang.keys())[0]
            first_sample_ids = train_sample_ids_by_lang[first_lang]
            for lang, sample_ids in train_sample_ids_by_lang.items():
                if sample_ids != first_sample_ids:
                    print(f"  ❌ Train set mismatch: {lang} differs from {first_lang}")
                    missing = first_sample_ids - sample_ids
                    extra = sample_ids - first_sample_ids
                    if missing:
                        print(f"    Missing in {lang}: {len(missing)} sample IDs")
                    if extra:
                        print(f"    Extra in {lang}: {len(extra)} sample IDs")
                else:
                    print(f"  ✅ Train set {lang} matches {first_lang}")
    else:
        print("  ⚠️  Train directory not found (--train flag not used)")
    
    # Test 5: Validation sets contain the same data across all languages
    print("\n5. Testing validation sets consistency...")
    if os.path.exists(val_dir):
        val_files = glob.glob(os.path.join(val_dir, "*.jsonl"))
        val_sample_ids_by_lang = {}
        
        for val_file in val_files:
            filename = os.path.basename(val_file)
            file_type, lang = get_file_type_and_lang(filename)
            if lang != "unknown":
                sample_ids = load_sample_ids(val_file)
                val_sample_ids_by_lang[lang] = sample_ids
                print(f"  {lang}: {len(sample_ids)} sample IDs")
        
        # Check if all validation sets have the same sample IDs
        if len(val_sample_ids_by_lang) > 1:
            first_lang = list(val_sample_ids_by_lang.keys())[0]
            first_sample_ids = val_sample_ids_by_lang[first_lang]
            for lang, sample_ids in val_sample_ids_by_lang.items():
                if sample_ids != first_sample_ids:
                    print(f"  ❌ Validation set mismatch: {lang} differs from {first_lang}")
                    missing = first_sample_ids - sample_ids
                    extra = sample_ids - first_sample_ids
                    if missing:
                        print(f"    Missing in {lang}: {len(missing)} sample IDs")
                    if extra:
                        print(f"    Extra in {lang}: {len(extra)} sample IDs")
                else:
                    print(f"  ✅ Validation set {lang} matches {first_lang}")
    else:
        print("  ⚠️  Validation directory not found (--train flag not used)")
    
    # Test 6: Train and validation sets don't overlap
    print("\n6. Testing train-validation overlap...")
    if os.path.exists(train_dir) and os.path.exists(val_dir) and train_sample_ids_by_lang and val_sample_ids_by_lang:
        first_train_lang = list(train_sample_ids_by_lang.keys())[0]
        first_val_lang = list(val_sample_ids_by_lang.keys())[0]
        train_sample_ids = train_sample_ids_by_lang[first_train_lang]
        val_sample_ids = val_sample_ids_by_lang[first_val_lang]
        
        overlap = train_sample_ids.intersection(val_sample_ids)
        if overlap:
            print(f"  ❌ Train and validation sets overlap: {len(overlap)} common sample IDs")
            print(f"    Overlapping sample IDs: {sorted(list(overlap))[:10]}...")
        else:
            print(f"  ✅ Train and validation sets are disjoint (no overlap)")
    
    # Test 7: Verify split sizes
    print("\n7. Testing split sizes...")
    if dev_sample_ids_by_lang and test_sample_ids_by_lang:
        first_dev_lang = list(dev_sample_ids_by_lang.keys())[0]
        first_test_lang = list(test_sample_ids_by_lang.keys())[0]
        dev_count = len(dev_sample_ids_by_lang[first_dev_lang])
        test_count = len(test_sample_ids_by_lang[first_test_lang])
        total_count = dev_count + test_count
        
        print(f"  Dev set: {dev_count} samples")
        print(f"  Test set: {test_count} samples")
        print(f"  Total: {total_count} samples")
        
        # Check if this matches expected split (75/25)
        expected_dev = int(total_count * 0.75)
        expected_test = total_count - expected_dev
        
        if dev_count == expected_dev and test_count == expected_test:
            print(f"  ✅ Split matches expected 75/25 ratio ({expected_dev}/{expected_test})")
        else:
            print(f"  ❌ Split doesn't match expected 75/25 ratio ({expected_dev}/{expected_test})")
    
    if os.path.exists(train_dir) and os.path.exists(val_dir) and train_sample_ids_by_lang and val_sample_ids_by_lang:
        first_train_lang = list(train_sample_ids_by_lang.keys())[0]
        first_val_lang = list(val_sample_ids_by_lang.keys())[0]
        train_count = len(train_sample_ids_by_lang[first_train_lang])
        val_count = len(val_sample_ids_by_lang[first_val_lang])
        
        print(f"  Train set: {train_count} samples")
        print(f"  Validation set: {val_count} samples")
        print(f"  Total dev: {train_count + val_count} samples")
        
        # Check if this matches expected split (80/20 of dev)
        if train_count == 134 and val_count == 34:
            print(f"  ✅ Train/val split matches expected 134/34 ratio")
        else:
            print(f"  ❌ Train/val split doesn't match expected 134/34 ratio")
    
    print("\n" + "=" * 50)
    print("Split validation complete!")

def main():
    parser = argparse.ArgumentParser(description='Test splits created by split_admin.py')
    parser.add_argument('--llm_inputs', action='store_true', help='Test LLM input splits')
    parser.add_argument('--short', action='store_true', help='Test short admin file splits')
    parser.add_argument('--base_dir', default='data/evaluation', help='Base directory for evaluation data')
    
    args = parser.parse_args()
    
    # Validate flags
    if args.llm_inputs and args.short:
        print("Error: Cannot use both --llm_inputs and --short flags")
        return
    
    # Run tests
    test_splits(args.base_dir, args.llm_inputs, args.short)

if __name__ == "__main__":
    main()
