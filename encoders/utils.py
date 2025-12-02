import torch
from transformers import PreTrainedTokenizer


def format_as_sequence_pair(example: dict, tokenizer: PreTrainedTokenizer) -> dict:
    """Format a single input example for token classification training.
    
    Args:
        example: Dictionary containing 'text_a', 'text_b', and 'labels_b' fields
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Dictionary containing 'input_ids', 'attention_mask', and 'labels'
    """
    # Split text_b into words and track their character spans
    text_b = example["text_b"]
    words_b = text_b.split()
    word_spans_b = []
    running_char_index = 0
    for word_index, word in enumerate(words_b):
        # Include preceding space in span, except for first word
        word_start = running_char_index if word_index == 0 else running_char_index - 1
        word_end = running_char_index + len(word)  # end is exclusive
        word_spans_b.append((word_start, word_end, word_index))
        running_char_index = word_end + 1  # skip the space

    # Tokenize text pair with offsets
    if tokenizer.model_name_or_path == "voidism/diffcse-roberta-base-sts":
        tokenized = tokenizer(
            example["text_a"],
            example["text_b"],
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
    else:
        tokenized = tokenizer(
            example["text_a"],
            example["text_b"],
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

    # Initialize labels with -100
    labels = torch.full_like(tokenized["input_ids"], fill_value=-100, dtype=torch.float32)
    
    # Find the start of the second sequence by looking for the second special token
    input_ids = tokenized["input_ids"][0].tolist()
    special_positions = [
        i for i, token_id in enumerate(input_ids)
        if token_id in tokenizer.all_special_ids
    ]

    if len(special_positions) < 2:
        raise ValueError("Could not find second special token in input_ids")
    seq_b_start = special_positions[1] + 1

    # Use offset_mapping (instead of word_ids) to map each token back to text_b
    offsets = tokenized["offset_mapping"][0].tolist()  # list of (start, end) tuples

    for i in range(seq_b_start, len(input_ids)):
        # Skip special/pad tokens
        if input_ids[i] in tokenizer.all_special_ids:
            continue

        start_char, end_char = offsets[i]
        # If there's no valid offset (e.g., (0, 0) for special tokens), skip
        if start_char == 0 and end_char == 0:
            continue

        # Find which word this token (or its start) belongs to in text_b
        # We use the token's start_char to determine its word
        for (w_start, w_end, w_index) in word_spans_b:
            if w_start <= start_char < w_end:
                if w_index < len(example["labels_b"]):
                    labels[0][i] = example["labels_b"][w_index]
                break

    # Replace any -1 labels with -100
    labels[labels == -1] = -100

    return {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0],
        "labels": labels[0]
    }


def ForTokenRegression(logits, labels, config=None, ignore_index: int = -100, **kwargs):
    """
    Loss function, adapted from transformers.loss.loss_utils.ForTokenClassification
    """
    logits = logits.view(-1)
    labels = labels.view(-1).to(logits.device)
    logits = logits.float()
    return torch.nn.BCEWithLogitsLoss()(logits[labels != ignore_index], labels[labels != ignore_index])


from transformers.loss.loss_utils import LOSS_MAPPING
LOSS_MAPPING["ForTokenRegression"] = ForTokenRegression
