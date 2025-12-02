#!/usr/bin/env python3
"""
Script to analyze sample ID mismatches between gold data, encoder predictions, and LLM predictions.
This script checks if the index of samples in files matches the number in their ID.

The script performs the following analyses:
1. Gold label files - checks sample ID consistency and sample counts
2. Encoder prediction files - checks sample ID consistency and sample counts  
3. LLM prediction files - checks sample ID consistency and sample counts
4. Gold vs Encoder predictions - compares sample counts, ID orders, and content
5. Gold vs LLM predictions - compares sample counts, ID orders, and content
6. LLM predictions vs LLM inputs - compares sample counts, ID orders, and content
"""

import json
import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import jsonlines
import argparse

def extract_id_number(sample_id: str) -> int:
    """Extract the numeric part from a sample ID (e.g., 'admin_de_123' -> 123)"""
    try:
        return int(sample_id.split('_')[-1])
    except (ValueError, IndexError):
        return -1

def analyze_file(file_path: str, is_llm_predictions: bool = False) -> Tuple[int, List[Tuple[int, str, bool]]]:
    """
    Analyze a single file and return:
    - Total number of samples
    - List of (index, id, id_matches_index) tuples
    """
    sample_count = 0
    id_analysis = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # Try both 'id' and 'item_id' fields (LLM predictions use 'item_id')
                    sample_id = data.get('id', '') or data.get('item_id', '')
                    
                    if sample_id:
                        sample_count += 1
                        id_number = extract_id_number(sample_id)
                        index_matches_id = (line_num == id_number)
                        id_analysis.append((line_num, sample_id, index_matches_id))
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, []
    
    return sample_count, id_analysis

def find_mismatches(id_analysis: List[Tuple[int, str, bool]]) -> List[Tuple[int, str, int]]:
    """Find samples where index doesn't match ID number"""
    mismatches = []
    for index, sample_id, matches in id_analysis:
        if not matches:
            id_number = extract_id_number(sample_id)
            mismatches.append((index, sample_id, id_number))
    return mismatches

def analyze_gold_files(gold_dir: str) -> Dict[str, Tuple[int, List[Tuple[int, str, bool]]]]:
    """Analyze all gold label files"""
    gold_files = glob.glob(os.path.join(gold_dir, "*.jsonl"))
    results = {}
    
    print("=" * 80)
    print("ANALYZING GOLD LABEL FILES")
    print("=" * 80)
    
    total_files = 0
    total_samples = 0
    total_mismatches = 0
    files_with_issues = []
    
    for file_path in sorted(gold_files):
        filename = os.path.basename(file_path)
        if not "admin" in filename:
            continue

        if 'backup' in filename or 'short' in filename:
            continue
        
        total_files += 1
        sample_count, id_analysis = analyze_file(file_path)
        results[filename] = (sample_count, id_analysis)
        total_samples += sample_count
        
        # Check for mismatches
        mismatches = find_mismatches(id_analysis)
        if mismatches:
            total_mismatches += len(mismatches)
            files_with_issues.append(filename)
            print(f"\nAnalyzing: {filename}")
            print(f"  Total samples: {sample_count}")
            print(f"  ❌ Found {len(mismatches)} ID mismatches:")
            for index, sample_id, id_number in mismatches[:5]:  # Show first 5
                print(f"    Index {index} -> ID '{sample_id}' (expected {id_number})")
            if len(mismatches) > 5:
                print(f"    ... and {len(mismatches) - 5} more")

    # Check sample count for admin files (should be 224)
    admin_files = [f for f in results.keys() if "admin" in f and "short" not in f]
    if admin_files:
        sample_count_issues = []
        for filename in admin_files:
            if 'backup' in filename or 'short' in filename:
                continue
            sample_count = results[filename][0]
            if sample_count != 224:
                sample_count_issues.append(f"{filename}: {sample_count} samples (expected 224)")
        
    # Print summary
    if files_with_issues:
        print(f"\n❌ GOLD FILES SUMMARY: {len(files_with_issues)} out of {total_files} files have issues")
        print(f"   Total mismatches: {total_mismatches}")
        #print(f"   Files with issues: {', '.join(files_with_issues)}")
    elif sample_count_issues:
        print(f"\n❌ SAMPLE COUNT ISSUES: {len(sample_count_issues)} admin files don't have 224 samples")
        for issue in sample_count_issues:
            print(f"   {issue}")
    else:
        print(f"\nGOLD FILES SUMMARY: All {total_files} files passed ({total_samples} total samples)")
        print(f" ✅   Tested: Sample ID numbers match their file indices")
        print(f" ✅   Tested: All {len(admin_files)} admin files have exactly 224 samples")
    

    return results

def analyze_prediction_files(pred_dir: str) -> Dict[str, Tuple[int, List[Tuple[int, str, bool]]]]:
    """Analyze all prediction files"""
    pred_files = glob.glob(os.path.join(pred_dir, "*.jsonl*"))
    results = {}
    
    print("\n" + "=" * 80)
    print("ANALYZING ENCODER PREDICTION FILES")
    print("=" * 80)
    
    total_files = 0
    total_samples = 0
    total_mismatches = 0
    files_with_issues = []
    
    for file_path in sorted(pred_files):
        filename = os.path.basename(file_path)
        if not "admin" in filename:
            continue

        if 'backup' in filename or 'short' in filename:
            continue
        
        total_files += 1
        sample_count, id_analysis = analyze_file(file_path)
        results[filename] = (sample_count, id_analysis)
        total_samples += sample_count
        
        # Check for mismatches
        mismatches = find_mismatches(id_analysis)
        if mismatches:
            total_mismatches += len(mismatches)
            files_with_issues.append(filename)
            print(f"\nAnalyzing: {filename}")
            print(f"  Total samples: {sample_count}")
            print(f"  ❌ Found {len(mismatches)} ID mismatches:")
            for index, sample_id, id_number in mismatches[:5]:  # Show first 5
                print(f"    Index {index} -> ID '{sample_id}' (expected {id_number})")
            if len(mismatches) > 5:
                print(f"    ... and {len(mismatches) - 5} more")

    # Check sample count for admin files (should be 224)
    admin_files = [f for f in results.keys() if "admin" in f and "short" not in f]
    if admin_files:
        sample_count_issues = []
        for filename in admin_files:
            if 'backup' in filename or 'short' in filename:
                continue
            sample_count = results[filename][0]
            if sample_count != 224:
                sample_count_issues.append(f"{filename}: {sample_count} samples (expected 224)")
    
    # Print summary
    if files_with_issues:
        print(f"\n❌ PREDICTION FILES SUMMARY: {len(files_with_issues)} out of {total_files} files have issues")
        print(f"   Total mismatches: {total_mismatches}")
        print(f"   Files with issues: {', '.join(files_with_issues)}")
    elif sample_count_issues:
        print(f"\n❌ SAMPLE COUNT ISSUES: {len(sample_count_issues)} admin files don't have 224 samples")
        for issue in sample_count_issues:
            print(f"   {issue}")
    else:
        print(f"\nPREDICTION FILES SUMMARY: All {total_files} files passed ({total_samples} total samples)")
        print(f" ✅   Tested: Sample ID numbers match their file indices")
        print(f" ✅   Tested: All {len(admin_files)} admin files have exactly 224 samples")
    

    return results

def analyze_llm_prediction_files(llm_pred_dir: str) -> Dict[str, Tuple[int, List[Tuple[int, str, bool]]]]:
    """Analyze all LLM prediction files"""
    llm_pred_files = glob.glob(os.path.join(llm_pred_dir, "*.jsonl*"))
    results = {}
    
    print("\n" + "=" * 80)
    print("ANALYZING LLM PREDICTION FILES")
    print("=" * 80)
    
    total_files = 0
    total_samples = 0
    total_mismatches = 0
    files_with_issues = []
    
    for file_path in sorted(llm_pred_files):
        filename = os.path.basename(file_path)
        if not "admin" in filename:
            continue

        if 'backup' in filename or 'short' in filename:
            continue
        
        total_files += 1
        sample_count, id_analysis = analyze_file(file_path, is_llm_predictions=True)
        results[filename] = (sample_count, id_analysis)
        total_samples += sample_count
        
        # Check for mismatches
        mismatches = find_mismatches(id_analysis)
        if mismatches:
            total_mismatches += len(mismatches)
            files_with_issues.append(filename)
            print(f"\nAnalyzing: {filename}")
            print(f"  Total samples: {sample_count}")
            print(f"  ❌ Found {len(mismatches)} ID mismatches:")
            for index, sample_id, id_number in mismatches[:5]:  # Show first 5
                print(f"    Index {index} -> ID '{sample_id}' (expected {id_number})")
            if len(mismatches) > 5:
                print(f"    ... and {len(mismatches) - 5} more")

    # Check sample count for admin files (should be 224)
    admin_files = [f for f in results.keys() if "admin" in f and "short" not in f]
    if admin_files:
        sample_count_issues = []
        for filename in admin_files:
            if 'backup' in filename or 'short' in filename:
                continue
            sample_count = results[filename][0]
            if sample_count != 224:
                sample_count_issues.append(f"{filename}: {sample_count} samples (expected 224)")
    
    # Print summary
    if files_with_issues:
        print(f"\n❌ LLM PREDICTION FILES SUMMARY: {len(files_with_issues)} out of {total_files} files have issues")
        print(f"   Total mismatches: {total_mismatches}")
        print(f"   Files with issues: {', '.join(files_with_issues)}")
    elif sample_count_issues:
        print(f"\n❌ SAMPLE COUNT ISSUES: {len(sample_count_issues)} admin files don't have 224 samples")
        for issue in sample_count_issues:
            print(f"   {issue}")
    else:
        print(f"\nLLM PREDICTION FILES SUMMARY: All {total_files} files passed ({total_samples} total samples)")
        print(f" ✅   Tested: Sample ID numbers match their file indices")
        print(f" ✅   Tested: All {len(admin_files)} admin files have exactly 224 samples")
    

    return results

def extract_sentences_from_llm_input(content: str) -> Tuple[str, str]:
    """Extract Sentence 1 and Sentence 2 from LLM input message content"""
    sentence1 = ""
    sentence2 = ""
    
    # Look for "Sentence 1:" and "Sentence 2:" in the content
    sentence1_match = re.search(r'Sentence 1: "(.*?)"(?:\n|$)', content, re.DOTALL)
    sentence2_match = re.search(r'Sentence 2: "(.*?)"(?:\n|$)', content, re.DOTALL)
    
    if sentence1_match:
        sentence1 = sentence1_match.group(1)
    if sentence2_match:
        sentence2 = sentence2_match.group(1)
    
    return sentence1, sentence2

def load_sample_data(file_path: str, is_llm_predictions: bool = False) -> Dict[str, Dict]:
    """Load sample data from a file, indexed by sample ID"""
    samples = {}
    try:
        with jsonlines.open(file_path) as reader:
            for item in reader:
                # Try both 'id' and 'item_id' fields (LLM predictions use 'item_id')
                sample_id = item.get('id', '') or item.get('item_id', '')
                if sample_id:
                    if is_llm_predictions and 'messages' in item:
                        # For LLM input files, extract sentences from message content
                        if item['messages'] and item['messages'][0]['role'] == 'user':
                            content = item['messages'][0]['content']
                            sentence1, sentence2 = extract_sentences_from_llm_input(content)
                            # Add extracted sentences to the item for comparison
                            item['extracted_sentence1'] = sentence1
                            item['extracted_sentence2'] = sentence2
                    samples[sample_id] = item
        return samples
    except Exception as e:
        print(f"Error loading sample data from {file_path}: {e}")
        return {}

def compare_gold_vs_predictions(gold_results: Dict, pred_results: Dict, split: str):
    """Compare gold files with corresponding prediction files"""
    print("\n" + "=" * 80)
    print("COMPARING GOLD VS ENCODER PREDICTIONS")
    print("=" * 80)
    
    # Find matching files
    gold_files = set(gold_results.keys())
    pred_files = set(pred_results.keys())
    
    total_comparisons = 0
    total_issues = 0
    files_with_issues = []
    
    # Look for files that might correspond to each other
    for gold_file in sorted(gold_files):
        if 'backup' in gold_file or 'short' in gold_file:
            continue
        gold_count, gold_analysis = gold_results[gold_file]
        
        # Find corresponding prediction files
        matching_preds = []
        for pred_file in pred_files:
            if 'backup' in pred_file or 'short' in pred_file:
                continue
            # Extract the _admin_lang pattern from both files
            gold_admin_match = re.search(r'_admin_([a-z]{2})', gold_file)
            pred_admin_match = re.search(r'_admin_([a-z]{2})', pred_file)
            
            if gold_admin_match and pred_admin_match:
                # Both files have _admin_lang pattern, check if language matches
                if gold_admin_match.group(1) == pred_admin_match.group(1):
                    matching_preds.append(pred_file)
            elif gold_file.replace('.jsonl', '') in pred_file or pred_file.replace('.jsonl', '') in gold_file:
                # Fallback to original substring matching for non-admin files
                matching_preds.append(pred_file)
        
        if matching_preds:
            for pred_file in matching_preds:
                total_comparisons += 1
                pred_count, pred_analysis = pred_results[pred_file]
                
                # Check for issues
                has_issues = False
                issues = []
                
                if gold_count != pred_count:
                    has_issues = True
                    issues.append(f"Sample count mismatch: Gold={gold_count}, Pred={pred_count}")
                
                # Check if IDs are in the same order
                gold_ids = [item[1] for item in gold_analysis]
                pred_ids = [item[1] for item in pred_analysis]
                
                if gold_ids != pred_ids:
                    has_issues = True
                    issues.append(f"Sample ID order differs")
                
                # Load sample data for content comparison
                gold_file_path = os.path.join("data/evaluation/gold_labels/{split}", gold_file)
                pred_file_path = os.path.join("data/evaluation/encoder_predictions/{split}", pred_file)
                
                gold_samples = load_sample_data(gold_file_path)
                pred_samples = load_sample_data(pred_file_path)
                
                # Check if sample content matches
                content_mismatches = []
                for gold_id in gold_ids:
                    if gold_id in gold_samples and gold_id in pred_samples:
                        gold_sample = gold_samples[gold_id]
                        pred_sample = pred_samples[gold_id]
                        
                        # Compare text_a, text_b, and id
                        if (gold_sample.get('text_a') != pred_sample.get('text_a') or
                            gold_sample.get('text_b') != pred_sample.get('text_b') or
                            gold_sample.get('id') != pred_sample.get('id')):
                            content_mismatches.append(gold_id)
                
                if content_mismatches:
                    has_issues = True
                    issues.append(f"Sample content mismatch: {len(content_mismatches)} samples differ")
                
                if has_issues:
                    total_issues += 1
                    files_with_issues.append((gold_file, pred_file, issues))
                    print(f"\nGold file: {gold_file}")
                    print(f"  Prediction file: {pred_file}")
                    print(f"    Gold samples: {gold_count}, Prediction samples: {pred_count}")
                    
                    if gold_count != pred_count:
                        print(f"    ❌ Sample count mismatch!")
                    else:
                        print(f"    ✅ Sample count matches")
                    
                    if gold_ids == pred_ids:
                        print(f"    ✅ Sample ID order matches")
                    else:
                        print(f"    ❌ Sample ID order differs!")
                        # Show first few differences
                        for i, (g_id, p_id) in enumerate(zip(gold_ids[:10], pred_ids[:10])):
                            if g_id != p_id:
                                print(f"      Index {i}: Gold='{g_id}' vs Pred='{p_id}'")
                    
                    if content_mismatches:
                        print(f"    ❌ Sample content mismatch!")
                        print(f"      {len(content_mismatches)} samples have different content")
                        # Show first few content mismatches
                        for sample_id in content_mismatches[:5]:
                            gold_sample = gold_samples[sample_id]
                            pred_sample = pred_samples[sample_id]
                            print(f"      Sample {sample_id}:")
                            if gold_sample.get('text_a') != pred_sample.get('text_a'):
                                print(f"        text_a differs")
                            if gold_sample.get('text_b') != pred_sample.get('text_b'):
                                print(f"        text_b differs")
                            if gold_sample.get('id') != pred_sample.get('id'):
                                print(f"        id differs")
                        if len(content_mismatches) > 5:
                            print(f"      ... and {len(content_mismatches) - 5} more content mismatches")
    
    # Print summary
    if files_with_issues:
        print(f"\n❌ COMPARISON SUMMARY: {total_issues} out of {total_comparisons} comparisons have issues")
        """print(f"   Files with issues:")
        for gold_file, pred_file, issues in files_with_issues:
            print(f"     {gold_file} vs {pred_file}: {', '.join(issues)}")"""
    else:
        print(f"\n✅ COMPARISON SUMMARY: All {total_comparisons} comparisons passed")
        print(f"   Tested: Sample counts, ID orders, and content (text_a, text_b, id) match between gold and prediction files")

def compare_gold_vs_llm_predictions(gold_results: Dict, llm_pred_results: Dict, split: str):
    """Compare gold files with corresponding LLM prediction files"""
    print("\n" + "=" * 80)
    print("COMPARING GOLD VS LLM PREDICTIONS")
    print("=" * 80)
    
    # Find matching files
    gold_files = set(gold_results.keys())
    llm_pred_files = set(llm_pred_results.keys())
    
    total_comparisons = 0
    total_issues = 0
    files_with_issues = []
    
    # Look for files that might correspond to each other
    for gold_file in sorted(gold_files):
        if 'backup' in gold_file or 'short' in gold_file:
            continue
        gold_count, gold_analysis = gold_results[gold_file]
        
        # Find corresponding LLM prediction files
        matching_llm_preds = []
        for llm_pred_file in llm_pred_files:
            if 'backup' in llm_pred_file or 'short' in llm_pred_file:
                continue
            # Extract the _admin_lang pattern from both files
            gold_admin_match = re.search(r'_admin_([a-z]{2})', gold_file)
            llm_pred_admin_match = re.search(r'_admin_([a-z]{2})', llm_pred_file)
            
            if gold_admin_match and llm_pred_admin_match:
                # Both files have _admin_lang pattern, check if language matches
                if gold_admin_match.group(1) == llm_pred_admin_match.group(1):
                    matching_llm_preds.append(llm_pred_file)
            elif gold_file.replace('.jsonl', '') in llm_pred_file or llm_pred_file.replace('.jsonl', '') in gold_file:
                # Fallback to original substring matching for non-admin files
                matching_llm_preds.append(llm_pred_file)
        
        if matching_llm_preds:
            for llm_pred_file in matching_llm_preds:
                total_comparisons += 1
                llm_pred_count, llm_pred_analysis = llm_pred_results[llm_pred_file]
                
                # Check for issues
                has_issues = False
                issues = []
                
                if gold_count != llm_pred_count:
                    has_issues = True
                    issues.append(f"Sample count mismatch: Gold={gold_count}, LLM Pred={llm_pred_count}")
                
                # Check if IDs are in the same order
                gold_ids = [item[1] for item in gold_analysis]
                llm_pred_ids = [item[1] for item in llm_pred_analysis]
                
                if gold_ids != llm_pred_ids:
                    has_issues = True
                    issues.append(f"Sample ID order differs")
                
                # Load sample data for content comparison
                gold_file_path = os.path.join(f"data/evaluation/gold_labels/{split}", gold_file)
                llm_pred_file_path = os.path.join(f"data/evaluation/llm_predictions/{split}", llm_pred_file)
                
                gold_samples = load_sample_data(gold_file_path)
                llm_pred_samples = load_sample_data(llm_pred_file_path, is_llm_predictions=True)
                
                # Check if sample content matches
                # COMMENTED OUT: Content comparison between gold and LLM predictions
                # content_mismatches = []
                # for gold_id in gold_ids:
                #     if gold_id in gold_samples and gold_id in llm_pred_samples:
                #         gold_sample = gold_samples[gold_id]
                #         llm_pred_sample = llm_pred_samples[gold_id]
                #         
                #         # Compare sentence1, sentence2, and id (LLM predictions use different field names)
                #         if (gold_sample.get('text_a') != llm_pred_sample.get('sentence1') or
                #             gold_sample.get('text_b') != llm_pred_sample.get('sentence2') or
                #             gold_sample.get('id') != llm_pred_sample.get('item_id')):
                #             content_mismatches.append(gold_id)
                
                # if content_mismatches:
                #     has_issues = True
                #     issues.append(f"Sample content mismatch: {len(content_mismatches)} samples differ")
                
                if has_issues:
                    total_issues += 1
                    files_with_issues.append((gold_file, llm_pred_file, issues))
                    print(f"\nGold file: {gold_file}")
                    print(f"  LLM Prediction file: {llm_pred_file}")
                    print(f"    Gold samples: {gold_count}, LLM Prediction samples: {llm_pred_count}")
                    
                    if gold_count != llm_pred_count:
                        print(f"    ❌ Sample count mismatch!")
                    else:
                        print(f"    ✅ Sample count matches")
                    
                    if gold_ids == llm_pred_ids:
                        print(f"    ✅ Sample ID order matches")
                    else:
                        print(f"    ❌ Sample ID order differs!")
                        # Show first few differences
                        for i, (g_id, p_id) in enumerate(zip(gold_ids[:10], llm_pred_ids[:10])):
                            if g_id != p_id:
                                print(f"      Index {i}: Gold='{g_id}' vs LLM Pred='{p_id}'")
                    
                    # COMMENTED OUT: Content mismatch display
                    # if content_mismatches:
                    #     print(f"    ❌ Sample content mismatch!")
                    #     print(f"      {len(content_mismatches)} samples have different content")
                    #     # Show first few content mismatches
                    #     for sample_id in content_mismatches[:5]:
                    #         gold_sample = gold_samples[sample_id]
                    #         llm_pred_sample = llm_pred_samples[sample_id]
                    #         print(f"      Sample {sample_id}:")
                    #         if gold_sample.get('text_a') != llm_pred_sample.get('sentence1'):
                    #             print(f"        text_a vs sentence1 differs")
                    #         if gold_sample.get('text_b') != llm_pred_sample.get('sentence2'):
                    #             print(f"        text_b vs sentence2 differs")
                    #         if gold_sample.get('id') != llm_pred_sample.get('item_id'):
                    #             print(f"        id vs item_id differs")
                    #     if len(content_mismatches) > 5:
                    #         print(f"      ... and {len(content_mismatches) - 5} more content mismatches")
    
    # Print summary
    if files_with_issues:
        print(f"\n❌ COMPARISON SUMMARY: {total_issues} out of {total_comparisons} comparisons have issues")
    else:
        print(f"\n✅ COMPARISON SUMMARY: All {total_comparisons} comparisons passed")
        print(f"   Tested: Sample counts and ID orders match between gold and LLM prediction files")
        print(f"   NOTE: Content comparison has been commented out")

def compare_llm_predictions_vs_inputs(llm_pred_results: Dict, llm_input_dir: str, split: str):
    """Compare LLM prediction files with corresponding LLM input files"""
    print("\n" + "=" * 80)
    print("COMPARING LLM PREDICTIONS VS LLM INPUTS")
    print("=" * 80)
    
    # Find matching files
    llm_pred_files = set(llm_pred_results.keys())
    
    # Get LLM input files
    if not os.path.exists(llm_input_dir):
        print(f"Warning: LLM input directory not found: {llm_input_dir}")
        return
    
    llm_input_files = glob.glob(os.path.join(llm_input_dir, "*.jsonl*"))
    llm_input_files = [os.path.basename(f) for f in llm_input_files if 'backup' not in f]
    
    total_comparisons = 0
    total_issues = 0
    files_with_issues = []
    
    # Look for files that might correspond to each other
    for llm_pred_file in sorted(llm_pred_files):
        if 'backup' in llm_pred_file or 'short' in llm_pred_file:
            continue
        llm_pred_count, llm_pred_analysis = llm_pred_results[llm_pred_file]
        
        # Find corresponding LLM input files
        matching_inputs = []
        for input_file in llm_input_files:
            if 'backup' in input_file or 'short' in input_file:
                continue
            # Extract the _admin_lang pattern from both files
            pred_admin_match = re.search(r'_admin_([a-z]{2})', llm_pred_file)
            input_admin_match = re.search(r'_admin_([a-z]{2})', input_file)
            
            if pred_admin_match and input_admin_match:
                # Both files have _admin_lang pattern, check if language matches
                if pred_admin_match.group(1) == input_admin_match.group(1):
                    matching_inputs.append(input_file)
            elif llm_pred_file.replace('.jsonl', '') in input_file or input_file.replace('.jsonl', '') in llm_pred_file:
                # Fallback to original substring matching for non-admin files
                matching_inputs.append(input_file)
        
        if matching_inputs:
            for input_file in matching_inputs:
                total_comparisons += 1
                
                # Load sample data for content comparison
                llm_pred_file_path = os.path.join(f"data/evaluation/llm_predictions/{split}", llm_pred_file)
                input_file_path = os.path.join(f"data/evaluation/llm_inputs/{split}", input_file)
                
                llm_pred_samples = load_sample_data(llm_pred_file_path, is_llm_predictions=False)
                input_samples = load_sample_data(input_file_path, is_llm_predictions=True)
                
                # Check for issues
                has_issues = False
                issues = []
                
                if len(llm_pred_samples) != len(input_samples):
                    has_issues = True
                    issues.append(f"Sample count mismatch: LLM Pred={len(llm_pred_samples)}, Input={len(input_samples)}")
                
                # Check if IDs are in the same order
                llm_pred_ids = list(llm_pred_samples.keys())
                input_ids = list(input_samples.keys())
                
                if llm_pred_ids != input_ids:
                    has_issues = True
                    issues.append(f"Sample ID order differs")
                
                # Check if sample content matches
                # COMMENTED OUT: Content comparison between LLM predictions and inputs
                # content_mismatches = []
                # for pred_id in llm_pred_ids:
                #     if pred_id in llm_pred_samples and pred_id in input_samples:
                #         pred_sample = llm_pred_samples[pred_id]
                #         input_sample = input_samples[pred_id]
                #         
                #         # Compare extracted sentences and id
                #         pred_sentence1 = pred_sample.get('extracted_sentence1', '')
                #         pred_sentence2 = pred_sample.get('extracted_sentence2', '')
                #         input_sentence1 = input_sample.get('extracted_sentence1', '')
                #         input_sentence2 = input_sample.get('extracted_sentence2', '')
                #         
                #         if (pred_sentence1 != input_sentence1 or
                #             pred_sentence2 != input_sentence2 or
                #             pred_sample.get('item_id') != input_sample.get('item_id')):
                #             content_mismatches.append(pred_id)
                
                # if content_mismatches:
                #     has_issues = True
                #     issues.append(f"Sample content mismatch: {len(content_mismatches)} samples differ")
                
                if has_issues:
                    total_issues += 1
                    files_with_issues.append((llm_pred_file, input_file, issues))
                    print(f"\nLLM Prediction file: {llm_pred_file}")
                    print(f"  LLM Input file: {input_file}")
                    print(f"    LLM Pred samples: {len(llm_pred_samples)}, Input samples: {len(input_samples)}")
                    
                    if len(llm_pred_samples) != len(input_samples):
                        print(f"    ❌ Sample count mismatch!")
                    else:
                        print(f"    ✅ Sample count matches")
                    
                    if llm_pred_ids == input_ids:
                        print(f"    ✅ Sample ID order matches")
                    else:
                        print(f"    ❌ Sample ID order differs!")
                        # Show first few differences
                        for i, (p_id, i_id) in enumerate(zip(llm_pred_ids[:10], input_ids[:10])):
                            if p_id != i_id:
                                print(f"      Index {i}: LLM Pred='{p_id}' vs Input='{i_id}'")
                    
                    # COMMENTED OUT: Content mismatch display
                    # if content_mismatches:
                    #     
                    #     print(f"    ❌ Sample content mismatch!")
                    #     print(f"      {len(content_mismatches)} samples have different content")
                    #     # Show first few content mismatches
                    #     for sample_id in content_mismatches[:5]:
                    #         pred_sample = llm_pred_samples[sample_id]
                    #         input_sample = input_samples[sample_id]
                    #         print(f"      Sample {sample_id}:")
                    #         
                    #         pred_sentence1 = pred_sample.get('sentence1', '')
                    #         pred_sentence2 = pred_sample.get('sentence2', '')
                    #         input_sentence1 = input_sample.get('extracted_sentence1', '')
                    #         input_sentence2 = input_sample.get('extracted_sentence2', '')
                    #         
                    #         if pred_sentence1 != input_sentence1:
                    #             print(f"        sentence1 differs:")
                    #             print(f"          Expected: {input_sentence1[:100]}{'...' if len(input_sentence1) > 100 else ''}")
                    #             print(f"          Got:      {pred_sentence1[:100]}{'...' if len(pred_sentence1) > 100 else ''}")
                    #         if pred_sentence2 != input_sentence2:
                    #             print(f"        sentence2 differs:")
                    #             print(f"          Expected: {input_sentence2[:100]}{'...' if len(input_sentence2) > 100 else ''}")
                    #             print(f"          Got:      {pred_sentence2[:100]}{'...' if len(pred_sentence2) > 100 else ''}")
                    #         if pred_sample.get('item_id') != input_sample.get('id'):
                    #             print(f"        item_id differs:")
                    #             print(f"          Expected: {input_sample.get('id')}")
                    #             print(f"          Got:      {pred_sample.get('item_id')}")
                    #     if len(content_mismatches) > 5:
                    #         print(f"      ... and {len(content_mismatches) - 5} more content mismatches")
    
    # Print summary
    if files_with_issues:
        print(f"\n❌ COMPARISON SUMMARY: {total_issues} out of {total_comparisons} comparisons have issues")
    else:
        print(f"\n✅ COMPARISON SUMMARY: All {total_comparisons} comparisons passed")
        print(f"   Tested: Sample counts and ID orders match between LLM predictions and LLM inputs")
        print(f"   NOTE: Content comparison has been commented out")

def main():
    parser = argparse.ArgumentParser(description='Test admin sample IDs')
    parser.add_argument('--split', type=str, help='Split to analyze', default="full")
    args = parser.parse_args()
    """Main analysis function"""
    gold_dir = f"data/evaluation/gold_labels/{args.split}"
    pred_dir = f"data/evaluation/encoder_predictions/{args.split}"
    llm_pred_dir = f"data/evaluation/llm_predictions/{args.split}"
    llm_input_dir = f"data/evaluation/llm_inputs/{args.split}"
    
    if not os.path.exists(gold_dir):
        print(f"Error: Gold directory not found: {gold_dir}")
        return
    
    if not os.path.exists(pred_dir):
        print(f"Error: Encoder prediction directory not found: {pred_dir}")
        return
    
    # Analyze gold files
    gold_results = analyze_gold_files(gold_dir)
    
    # Analyze encoder prediction files
    pred_results = analyze_prediction_files(pred_dir)
    
    # Analyze LLM prediction files
    llm_pred_results = {}
    if os.path.exists(llm_pred_dir):
        llm_pred_results = analyze_llm_prediction_files(llm_pred_dir)
    else:
        print(f"\nWarning: LLM predictions directory not found: {llm_pred_dir}")
    
    # Compare results
    compare_gold_vs_predictions(gold_results, pred_results, args.split)
    
    if llm_pred_results:
        # Compare gold vs LLM predictions
        compare_gold_vs_llm_predictions(gold_results, llm_pred_results, args.split)
        
        # Compare LLM predictions vs LLM inputs
        if os.path.exists(llm_input_dir):
            compare_llm_predictions_vs_inputs(llm_pred_results, llm_input_dir, args.split)
        else:
            print(f"\nWarning: LLM inputs directory not found: {llm_input_dir}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
