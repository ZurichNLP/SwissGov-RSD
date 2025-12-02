import jsonlines
import random
import glob
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Split admin, LLM input, or prediction files into dev and test sets')
    parser.add_argument('--llm_inputs', action='store_true', help='Split LLM input files instead of admin files')
    parser.add_argument('--predictions', action='store_true', help='Split existing prediction files (assumes same structure as admin files) instead of admin files')
    parser.add_argument('--short', action='store_true', help='Process short admin files (not applicable for LLM inputs or predictions)')
    parser.add_argument('--train', action='store_true', help='Split dev set into train and validation sets (134/34 split)')
    parser.add_argument('--prefix', type=str, help='Custom file prefix to filter files (e.g., "llama-405b-0-236-admin-" or "gold_admin_")')
    args = parser.parse_args()
    
    # Validate flags
    if args.short and args.llm_inputs:
        raise ValueError("--short flag is not applicable for LLM inputs")
    
    if args.llm_inputs and args.predictions:
        raise ValueError("Cannot use both --llm_inputs and --predictions flags")
    
    # --prefix can be used with predictions, short admin files, or regular admin files
    # (but not with LLM inputs)
    
    ids = range(0,224)
    
    # randomly split the ids into two lists that contain 75 and 25 % of the ids
    # Convert to list to allow shuffling
    # Set random seed for reproducibility
    random.seed(42)
    
    ids_list = list(ids)
    random.shuffle(ids_list)
    
    # Split into non-overlapping sets
    split_point = int(len(ids_list) * 0.75)
    dev_ids = ids_list[:split_point]
    test_ids = ids_list[split_point:]
    
    # Verify no overlap
    assert len(set(dev_ids).intersection(set(test_ids))) == 0
    assert len(dev_ids) + len(test_ids) == len(ids)
    assert set(dev_ids).union(set(test_ids)) == set(ids)

    # turn list of numbers into proper ids
    dev_ids_de = [f"admin_de_{id}" for id in dev_ids]
    test_ids_de = [f"admin_de_{id}" for id in test_ids]

    dev_ids_fr = [f"admin_fr_{id}" for id in dev_ids]
    test_ids_fr = [f"admin_fr_{id}" for id in test_ids]

    dev_ids_it = [f"admin_it_{id}" for id in dev_ids]
    test_ids_it = [f"admin_it_{id}" for id in test_ids]

    assert len(dev_ids_de) == len(dev_ids_fr) == len(dev_ids_it)
    assert len(test_ids_de) == len(test_ids_fr) == len(test_ids_it)

    assert len(dev_ids_de) + len(test_ids_de) == len(dev_ids_fr) + len(test_ids_fr) == len(dev_ids_it) + len(test_ids_it) == len(ids)

    if args.llm_inputs:
        dev_dir = "data/evaluation/llm_inputs/dev"
        test_dir = "data/evaluation/llm_inputs/test"
        print("Processing LLM input files...")
    elif args.predictions:
        if "encoder" in str(args.prefix):
            dev_dir = f"data/evaluation/encoder_predictions/dev/"
            test_dir = f"data/evaluation/encoder_predictions/test/"
            print("Processing prediction files with custom prefix...")
        else:
            dev_dir = f"data/evaluation/llm_predictions/dev/"
            test_dir = f"data/evaluation/llm_predictions/test/"
            print("Processing prediction files...")
    else:
        dev_dir = "data/evaluation/gold_labels/dev"
        test_dir = "data/evaluation/gold_labels/test"
        print("Processing admin files...")
    
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Determine which files to process based on the flag
    if args.llm_inputs:
        # Process LLM input files
        input_files = glob.glob("data/evaluation/llm_inputs/full/*admin*.jsonl")
        file_pattern = r"test_admin_([a-z]{2})"  # Extract language code from LLM input filenames
    elif args.predictions:
        # Process prediction files
        if args.prefix:
            # Use custom prefix to filter files
            if args.short:
                # For short files, look for files with _short in the name
                input_files = glob.glob(f"{args.prefix}*_short*")
                print(f"Using custom prefix for short prediction files: {args.prefix}")
            else:
                # For regular files, exclude short files
                input_files = glob.glob(f"{args.prefix}*")
                input_files = [f for f in input_files if '_short' not in f]
                print(f"Using custom prefix for regular prediction files: {args.prefix}")
        else:
            # Use default pattern to find all admin files
            if args.short:
                # For short files, look for files with _short in the name
                input_files = glob.glob("data/evaluation/llm_predictions/full/*_short*")
            else:
                # For regular files, exclude short files
                input_files = glob.glob("data/evaluation/llm_predictions/full/*.json*")
                input_files = [f for f in input_files if '_short' not in f]
            
            
        # Use a more flexible pattern that matches admin_*, *_admin_*, admin-*, and _*_short formats
        if args.short:
            file_pattern = r"(?:.*?)_([a-z]{2})_short"  # Extract language code from short prediction filenames
        else:
            file_pattern = r"(?:.*?)(?:admin_|_admin_|admin-)([a-z]{2})"  # Extract language code from prediction filenames
        print(f"Found {len(input_files)} prediction files with language codes")
    else:
        # Process admin files
        if args.short:
            if args.prefix:
                # Use custom prefix to filter short admin files
                input_files = glob.glob(f"{args.prefix}*short*.jsonl")
                print(f"Using custom prefix for short files: {args.prefix}")
            else:
                input_files = glob.glob("data/evaluation/gold_labels/*admin*short*.jsonl")
            file_pattern = r"gold_admin_([a-z]{2})_short"  # Extract language code from short admin filenames
        else:
            # For regular admin files, exclude short files
            if args.prefix:
                # Use custom prefix to filter regular admin files
                input_files = glob.glob(f"{args.prefix}*.jsonl")
                # Filter out short files
                input_files = [f for f in input_files if 'short' not in f]
                print(f"Using custom prefix for regular admin files: {args.prefix}")
            else:
                all_admin_files = glob.glob("data/evaluation/gold_labels/*admin*.jsonl")
                input_files = [f for f in all_admin_files if 'short' not in f]
            file_pattern = r"admin_([a-z]{2})"  # Extract language code from admin filenames
    
    # Clear existing files in dev and test directories
    for file in input_files:
        filename = file.split('/')[-1]
        dev_file = os.path.join(dev_dir, filename)
        test_file = os.path.join(test_dir, filename)
        
        # Clear existing files
        if os.path.exists(dev_file):
            os.remove(dev_file)
        if os.path.exists(test_file):
            os.remove(test_file)
    
    for file in input_files:
        filename = file.split('/')[-1]
        dev_file = os.path.join(dev_dir, filename)
        test_file = os.path.join(test_dir, filename)
        # Extract language code using regex pattern matching
        import re
        lang = re.search(file_pattern, file).group(1)
        devids = dev_ids_de if lang == "de" else dev_ids_fr if lang == "fr" else dev_ids_it
        testids = test_ids_de if lang == "de" else test_ids_fr if lang == "fr" else test_ids_it
        with jsonlines.open(file) as f:
            for line in f:
                # For predictions, use item_id instead of id
                line_id = line.get("item_id", line.get("id"))
                if line_id in devids:
                    #print(f"Writing: {line_id} to dev")
                    with jsonlines.open(dev_file, 'a') as dev_f:
                        dev_f.write(line)
                elif line_id in testids:
                    #print(f"Writing: {line_id} to test")
                    with jsonlines.open(test_file, 'a') as test_f:
                        test_f.write(line)
        
        # Verify the split for this file
        with jsonlines.open(f"{dev_file}", 'r') as dev_f:
            dev_lines = list(dev_f)
        with jsonlines.open(f"{test_file}", 'r') as test_f:
            test_lines = list(test_f)
        
        if args.short:
            # For short files, verify unique IDs instead of total counts
            dev_unique_ids = [line.get("item_id", line.get("id")) for line in dev_lines]
            test_unique_ids = [line.get("item_id", line.get("id")) for line in test_lines]
            
            # Check that all expected IDs are present
            devids_set = set(devids)
            testids_set = set(testids)
            
            assert devids_set.issubset(dev_unique_ids), f"Dev split missing IDs for {file}: missing {devids_set - dev_unique_ids}"
            assert testids_set.issubset(test_unique_ids), f"Test split missing IDs for {file}: missing {testids_set - test_unique_ids}"
            
            print(f"Short file {file}: dev has {len(dev_lines)} lines with {len(dev_unique_ids)} unique IDs, test has {len(test_lines)} lines with {len(test_unique_ids)} unique IDs")
        else:
            # For regular files, verify total counts
            assert len(dev_lines) == len(devids), f"Dev split mismatch for {file}: expected {len(devids)}, got {len(dev_lines)}"
            assert len(test_lines) == len(testids), f"Test split mismatch for {file}: expected {len(testids)}, got {len(test_lines)}"
            assert len(dev_lines) + len(test_lines) == len(ids), f"Total split mismatch for {file}: expected {len(ids)}, got {len(dev_lines) + len(test_lines)}"

    if args.llm_inputs:
        file_type = "LLM input"
    elif args.predictions:
        file_type = "prediction"
    elif args.short:
        file_type = "short admin"
    else:
        file_type = "admin"
    
    print(f"Successfully split {len(input_files)} {file_type} files:")
    print(f"  Dev set: {len(dev_ids)} document IDs per language ({len(dev_ids) * 3} total)")
    print(f"  Test set: {len(test_ids)} document IDs per language ({len(test_ids) * 3} total)")
    print(f"  Total documents: {len(ids)} per language ({len(ids) * 3} total)")

    #.jsonl Split dev set into train and validation if --train flag is set
    if args.train:
        print("\nSplitting dev set into train and validation sets...")
        
        # Create train and validation directories
        train_dir = os.path.join(dev_dir, "train")
        val_dir = os.path.join(dev_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Split dev IDs into train (134) and validation (34)
        train_ids = dev_ids[:134]
        val_ids = dev_ids[134:]
        
        # Create train and validation ID lists for each language
        train_ids_de = [f"admin_de_{id}" for id in train_ids]
        val_ids_de = [f"admin_de_{id}" for id in val_ids]
        train_ids_fr = [f"admin_fr_{id}" for id in train_ids]
        val_ids_fr = [f"admin_fr_{id}" for id in val_ids]
        train_ids_it = [f"admin_it_{id}" for id in train_ids]
        val_ids_it = [f"admin_it_{id}" for id in val_ids]
        
        # Verify the split
        assert len(train_ids) == 134, f"Train split mismatch: expected 134, got {len(train_ids)}"
        assert len(val_ids) == 34, f"Validation split mismatch: expected 34, got {len(val_ids)}"
        assert len(train_ids) + len(val_ids) == len(dev_ids), f"Train/val split mismatch: expected {len(dev_ids)}, got {len(train_ids) + len(val_ids)}"
        
        # Process each dev file to create train and validation splits
        if args.prefix:
            if args.predictions:
                # Use custom prefix to filter dev files for predictions
                if args.short:
                    dev_files = glob.glob(f"{dev_dir}/{args.prefix}*_short*.jsonl")
                else:
                    dev_files = glob.glob(f"{dev_dir}/{args.prefix}*")
                    # Filter out short files
                    dev_files = [f for f in dev_files if '_short' not in f]
            else:
                # Use custom prefix to filter dev files for admin files
                if args.short:
                    dev_files = glob.glob(f"{dev_dir}/{args.prefix}*short*.jsonl")
                else:
                    dev_files = glob.glob(f"{dev_dir}/{args.prefix}*.jsonl")
                    # Filter out short files
                    dev_files = [f for f in dev_files if 'short' not in f]
        else:
            dev_files = glob.glob(f"{dev_dir}/*.jsonl")
        
        # Filter out short files if we're not in short mode
        if not args.short:
            dev_files = [f for f in dev_files if 'short' not in f]
        
        for dev_file in dev_files:
            filename = dev_file.split('/')[-1]
            train_file = os.path.join(train_dir, filename)
            val_file = os.path.join(val_dir, filename)
            
            # Clear existing files
            if os.path.exists(train_file):
                os.remove(train_file)
            if os.path.exists(val_file):
                os.remove(val_file)
            
            # Determine language and corresponding ID lists
            if args.llm_inputs:
                lang = re.search(r"test_admin_([a-z]{2})", filename).group(1)
            elif args.predictions:
                if args.short:
                    lang = re.search(r"(?:.*?)_([a-z]{2})_short", filename).group(1)
                else:
                    lang = re.search(r"(?:.*?)(?:admin_|_admin_|admin-)([a-z]{2})", filename).group(1)
            elif args.short:
                # Debug: print filename and regex match
                print(f"Debug: Processing filename '{filename}' with short pattern")
                match = re.search(r"gold_admin_([a-z]{2})_short", filename)
                if match is None:
                    print(f"Error: Regex pattern 'gold_admin_([a-z]{2})_short' did not match filename '{filename}'")
                    # Try alternative patterns
                    alt_match = re.search(r"admin_([a-z]{2})", filename)
                    if alt_match:
                        print(f"Alternative pattern 'admin_([a-z]{2})' matched: {alt_match.group(1)}")
                        lang = alt_match.group(1)
                    else:
                        raise ValueError(f"Could not extract language from filename '{filename}'")
                else:
                    lang = match.group(1)
            else:
                lang = re.search(r"admin_([a-z]{2})", filename).group(1)
            
            train_ids_lang = train_ids_de if lang == "de" else train_ids_fr if lang == "fr" else train_ids_it
            val_ids_lang = val_ids_de if lang == "de" else val_ids_fr if lang == "fr" else val_ids_it
            
            # Read dev file and split into train and validation
            with jsonlines.open(dev_file, 'r') as f:
                for line in f:
                    # For predictions, use item_id instead of id
                    line_id = line.get("item_id", line.get("id"))
                    if line_id in train_ids_lang:
                        with jsonlines.open(train_file, 'a') as train_f:
                            train_f.write(line)
                    elif line_id in val_ids_lang:
                        with jsonlines.open(val_file, 'a') as val_f:
                            val_f.write(line)
            
            # Verify the train/val split for this file
            with jsonlines.open(train_file, 'r') as train_f:
                train_lines = list(train_f)
            with jsonlines.open(val_file, 'r') as val_f:
                val_lines = list(val_f)
            
            if args.short:
                # For short files, verify unique IDs
                train_unique_ids = set(line.get("item_id", line.get("id")) for line in train_lines)
                val_unique_ids = set(line.get("item_id", line.get("id")) for line in val_lines)
                
                train_ids_set = set(train_ids_lang)
                val_ids_set = set(val_ids_lang)
                
                assert train_ids_set.issubset(train_unique_ids), f"Train split missing IDs for {filename}: missing {train_ids_set - train_unique_ids}"
                assert val_ids_set.issubset(val_unique_ids), f"Validation split missing IDs for {filename}: missing {val_ids_set - val_unique_ids}"
                
                print(f"  {filename}: train has {len(train_lines)} lines with {len(train_unique_ids)} unique IDs, val has {len(val_lines)} lines with {len(val_unique_ids)} unique IDs")
            else:
                # For regular files, verify counts
                assert len(train_lines) == len(train_ids_lang), f"Train split mismatch for {filename}: expected {len(train_ids_lang)}, got {len(train_lines)}"
                assert len(val_lines) == len(val_ids_lang), f"Validation split mismatch for {filename}: expected {len(val_ids_lang)}, got {len(val_lines)}"
                assert len(train_lines) + len(val_lines) == len(dev_ids), f"Train/val total mismatch for {filename}: expected {len(dev_ids)}, got {len(train_lines) + len(val_lines)}"
        
        print(f"Successfully created train/validation split:")
        print(f"  Train set: {len(train_ids)} document IDs per language ({len(train_ids) * 3} total)")
        print(f"  Validation set: {len(val_ids)} document IDs per language ({len(val_ids) * 3} total)")
        print(f"  Total dev documents: {len(dev_ids)} per language ({len(dev_ids) * 3} total)")

if __name__ == "__main__":
    main()