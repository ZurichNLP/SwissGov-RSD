#!/usr/bin/env python3
"""
Script to analyze sample ID consistency in short/split text gold data.
This script checks:
1. Whether there are exactly 224 documents (not chunks)
2. Whether chunk IDs are consecutive within each document
3. Whether document IDs are consecutive
4. Whether the structure is consistent

The key insight is that one line = one chunk, not one document.
Multiple chunks can belong to the same document.
"""

import json
import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import jsonlines
from collections import defaultdict

def extract_document_number(sample_id: str) -> int:
    """Extract the document number from a sample ID (e.g., 'admin_de_123' -> 123)"""
    try:
        return int(sample_id.split('_')[-1])
    except (ValueError, IndexError):
        return -1

def analyze_short_gold_file(file_path: str) -> Dict:
    """
    Analyze a single short gold label file and return:
    - Total number of chunks
    - Total number of unique documents
    - Document structure analysis
    """
    chunks = []
    documents = defaultdict(list)
    
    try:
        with jsonlines.open(file_path) as reader:
            for line_num, item in enumerate(reader):
                chunks.append(item)
                
                # Group by document ID
                doc_id = item.get('id', '')
                if doc_id:
                    documents[doc_id].append(item)
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}
    
    # Analyze document structure
    doc_analysis = {}
    for doc_id, doc_chunks in documents.items():
        doc_num = extract_document_number(doc_id)
        
        # Check chunk IDs within this document
        chunk_ids = [chunk.get('chunk_id', -1) for chunk in doc_chunks]
        chunk_ids.sort()
        
        # Check if chunk IDs are consecutive starting from 0
        expected_chunk_ids = list(range(len(chunk_ids)))
        chunk_ids_consecutive = chunk_ids == expected_chunk_ids
        
        # Check if all chunks have the same page_en
        page_en = doc_chunks[0].get('page_en', '')
        all_same_page = all(chunk.get('page_en', '') == page_en for chunk in doc_chunks)
        
        doc_analysis[doc_id] = {
            'doc_number': doc_num,
            'chunk_count': len(doc_chunks),
            'chunk_ids': chunk_ids,
            'chunk_ids_consecutive': chunk_ids_consecutive,
            'page_en': page_en,
            'all_same_page': all_same_page,
            'chunks': doc_chunks
        }
    
    return {
        'total_chunks': len(chunks),
        'total_documents': len(documents),
        'documents': doc_analysis,
        'chunks': chunks
    }

def analyze_short_gold_files(gold_dir: str) -> Dict[str, Dict]:
    """Analyze all short gold label files"""
    gold_files = glob.glob(os.path.join(gold_dir, "*.jsonl"))
    results = {}
    
    print("=" * 80)
    print("ANALYZING SHORT GOLD LABEL FILES")
    print("=" * 80)
    
    total_files = 0
    total_chunks = 0
    total_documents = 0
    files_with_issues = []
    
    for file_path in sorted(gold_files):
        filename = os.path.basename(file_path)
        if not "admin" in filename or "short" not in filename:
            continue

        if 'backup' in filename:
            continue
        
        total_files += 1
        analysis = analyze_short_gold_file(file_path)
        results[filename] = analysis
        total_chunks += analysis['total_chunks']
        total_documents += analysis['total_documents']
        
        # Check for issues
        issues = []
        
        # Check if we have exactly 224 documents
        if analysis['total_documents'] != 224:
            issues.append(f"Expected 224 documents, got {analysis['total_documents']}")
        
        # Check document ID consecutiveness
        doc_numbers = [analysis['documents'][doc_id]['doc_number'] for doc_id in analysis['documents']]
        doc_numbers.sort()
        expected_doc_numbers = list(range(0, len(doc_numbers)))  # Should start from 0 and go to len-1
        doc_ids_consecutive = doc_numbers == expected_doc_numbers
        
        if not doc_ids_consecutive:
            # Find specific gaps in document IDs
            gaps = []
            for i in range(len(doc_numbers) - 1):
                if doc_numbers[i + 1] != doc_numbers[i] + 1:
                    gaps.append(f"Gap between {doc_numbers[i]} and {doc_numbers[i + 1]}")
            
            # Find missing document IDs
            missing_ids = []
            for expected_id in expected_doc_numbers:
                if expected_id not in doc_numbers:
                    missing_ids.append(expected_id)
            
            # Find extra/unexpected document IDs
            extra_ids = []
            for doc_id in doc_numbers:
                if doc_id not in expected_doc_numbers:
                    extra_ids.append(doc_id)
            
            gap_details = []
            if gaps:
                gap_details.append(f"Gaps: {', '.join(gaps[:5])}{'...' if len(gaps) > 5 else ''}")
            if missing_ids:
                gap_details.append(f"Missing IDs: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")
            if extra_ids:
                gap_details.append(f"Extra IDs: {extra_ids[:10]}{'...' if len(extra_ids) > 10 else ''}")
            
            issues.append(f"Document IDs are not consecutive: {'; '.join(gap_details)}")
        
        # Check chunk ID consecutiveness within documents
        chunk_issues = []
        for doc_id, doc_info in analysis['documents'].items():
            if not doc_info['chunk_ids_consecutive']:
                chunk_issues.append(f"{doc_id}: chunk_ids {doc_info['chunk_ids']} not consecutive")
            if not doc_info['all_same_page']:
                chunk_issues.append(f"{doc_id}: chunks have different page_en values")
        
        if chunk_issues:
            issues.extend(chunk_issues[:5])  # Show first 5 chunk issues
            if len(chunk_issues) > 5:
                issues.append(f"... and {len(chunk_issues) - 5} more chunk issues")
        
        if issues:
            files_with_issues.append((filename, issues))
            print(f"\nAnalyzing: {filename}")
            print(f"  Total chunks: {analysis['total_chunks']}")
            print(f"  Total documents: {analysis['total_documents']}")
            print(f"  ❌ Found {len(issues)} issues:")
            for issue in issues:
                print(f"    {issue}")
        else:
            print(f"\n✅ {filename}: {analysis['total_chunks']} chunks, {analysis['total_documents']} documents")

    # Print summary
    if files_with_issues:
        print(f"\n❌ SHORT GOLD FILES SUMMARY: {len(files_with_issues)} out of {total_files} files have issues")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Total documents: {total_documents}")
        print(f"   Expected documents: 224")
    else:
        print(f"\n✅ SHORT GOLD FILES SUMMARY: All {total_files} files passed")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Total documents: {total_documents}")
        print(f"   ✅ All files have exactly 224 documents")
        print(f"   ✅ All chunk IDs are consecutive within documents")
        print(f"   ✅ All document IDs are consecutive")

    return results


def main():
    """Main analysis function"""
    gold_dir = "data/evaluation/gold_labels/full"
    
    if not os.path.exists(gold_dir):
        print(f"Error: Gold directory not found: {gold_dir}")
        return
    
    # Analyze short gold files
    results = analyze_short_gold_files(gold_dir)
    
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
