import sys
import os
from pathlib import Path
import json


from copy import deepcopy
from pathlib import Path

import jsonlines
import jinja2
from datasets import concatenate_datasets, load_dataset
import argparse as ap

# Get the absolute path to the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the root dir to sys.path so Python can find the 'rsd' package
sys.path.insert(0, ROOT_DIR)

from rsd.data.ists import ISTSDataset
from rsd.data.pawsx import PAWSXDataset
from rsd.experiments.benchmark import MultiLengthDifferenceRecognitionBenchmark
from scripts.project_labels import project_labels



NUM_TRAIN_SEED_ANNOTATIONS = 2800
NUM_VALID_SEED_ANNOTATIONS = 200
MAX_ENCODER_TRAIN_INPUTS = 10000
MAX_ENCODER_VALID_INPUTS = 200
MAX_LLM_TRAIN_INPUTS = 560
MAX_LLM_VALID_INPUTS = 200

POSITIVE_RATIO = 0.5
MAX_SENTENCES_PER_DOCUMENT = 5
MAX_INVERSIONS = 5

LLM_LABELS_MAP = {
    0.0: 5,
    0.2: 4,
    0.4: 3,
    0.6: 2,
    0.8: 1,
    1.0: 0,
    -1.0: -1,
}

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--lang", type=str, help="Language to use for training data")
    parser.add_argument("--sents", action="store_true", help="If used, this will be sentence-level data only. Does include negative sentence-level data from PAWSX.")
    parser.add_argument("--multiling", action="store_true", help="If used, the training data will be multilingual/cross-lingual.")
    parser.add_argument("--multiling_large", type=int, help="If used, the training data will be multilingual/cross-lingual with 7 times more data.")
    parser.add_argument("--project_labels", type=bool, default=False, help="If used, the labels will be projected using LLM.")
    args = parser.parse_args()

    if args.sents:
        MAX_SENTENCES_PER_DOCUMENT = 1
        MAX_INVERSIONS = 0
        MAX_ENCODER_TRAIN_INPUTS = 3000
        NUM_TRAIN_SEED_ANNOTATIONS = NUM_TRAIN_SEED_ANNOTATIONS*2 if not args.lang == "en" else NUM_TRAIN_SEED_ANNOTATIONS
        NUM_VALID_SEED_ANNOTATIONS = NUM_VALID_SEED_ANNOTATIONS*2 if not args.lang == "en" else NUM_VALID_SEED_ANNOTATIONS
    elif args.lang or args.multiling:
        NUM_TRAIN_SEED_ANNOTATIONS = NUM_TRAIN_SEED_ANNOTATIONS*2 if not args.lang == "en" else NUM_TRAIN_SEED_ANNOTATIONS
        NUM_VALID_SEED_ANNOTATIONS = NUM_VALID_SEED_ANNOTATIONS*2 if not args.lang == "en" else NUM_VALID_SEED_ANNOTATIONS

    # if args.multiling_large:
    #     MAX_ENCODER_TRAIN_INPUTS = MAX_ENCODER_TRAIN_INPUTS*args.multiling_large
    #     MAX_ENCODER_VALID_INPUTS = MAX_ENCODER_VALID_INPUTS*args.multiling_large
    #     MAX_LLM_TRAIN_INPUTS = MAX_LLM_TRAIN_INPUTS*args.multiling_large
    #     MAX_LLM_VALID_INPUTS = MAX_LLM_VALID_INPUTS*args.multiling_large
    

    if args.lang and args.lang != "en":
        prompt_template_path = Path(__file__).parent.parent / "prompt_templates" / f"template_{args.lang}.txt"
    else:
        prompt_template_path = Path(__file__).parent.parent / "prompt_templates" / "template.txt"
    prompt_template = jinja2.Template(prompt_template_path.read_text())

    data_dir = Path(__file__).parent.parent / "data"
    if args.sents and not args.multiling and not args.project_labels:
        train_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"train_{args.lang}_sents.jsonl"
        valid_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"valid_{args.lang}_sents.jsonl"
        train_llm_path = data_dir / "finetuning" / "llm_inputs" / f"train_{args.lang}_sents.jsonl"
        valid_llm_path = data_dir / "finetuning" / "llm_inputs" / f"valid_{args.lang}_sents.jsonl"
    elif args.sents and args.multiling:
        train_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"train_multilingual_sents.jsonl"
        valid_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"valid_multilingual_sents.jsonl"
        train_llm_path = data_dir / "finetuning" / "llm_inputs" / f"train_multilingual_sents.jsonl"
        valid_llm_path = data_dir / "finetuning" / "llm_inputs" / f"valid_multilingual_sents.jsonl"
    elif args.multiling:
        train_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"train_multilingual.jsonl"
        valid_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"valid_multilingual.jsonl"
        train_llm_path = data_dir / "finetuning" / "llm_inputs" / f"train_multilingual.jsonl"
        valid_llm_path = data_dir / "finetuning" / "llm_inputs" / f"valid_multilingual.jsonl"
    elif args.multiling_large:
        train_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"train_multilingual_large.jsonl"
        valid_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"valid_multilingual_large.jsonl"
        train_llm_path = data_dir / "finetuning" / "llm_inputs" / f"train_multilingual_large.jsonl"
        valid_llm_path = data_dir / "finetuning" / "llm_inputs" / f"valid_multilingual_large.jsonl"
    elif args.project_labels:
        train_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"train_{args.lang}_projected.jsonl"
        valid_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"valid_{args.lang}_projected.jsonl"
        train_llm_path = data_dir / "finetuning" / "llm_inputs" / f"train_{args.lang}_projected.jsonl"
        valid_llm_path = data_dir / "finetuning" / "llm_inputs" / f"valid_{args.lang}_projected.jsonl"
    else:
        train_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"train_{args.lang}.jsonl"
        valid_encoder_path = data_dir / "finetuning" / "encoder_inputs" / f"valid_{args.lang}.jsonl"
        train_llm_path = data_dir / "finetuning" / "llm_inputs" / f"train_{args.lang}.jsonl"
        valid_llm_path = data_dir / "finetuning" / "llm_inputs" / f"valid_{args.lang}.jsonl"

    if not args.multiling and not args.multiling_large:
        positive_dataset = ISTSDataset(split="train", tgt_lang=args.lang)
        positive_dataset.dataset = positive_dataset.dataset.shuffle(seed=42)
        negative_dataset = PAWSXDataset(split="train", language=args.lang)
        negative_dataset.dataset = negative_dataset.dataset.shuffle(seed=42)

        print(len(positive_dataset.dataset))
        print(len(negative_dataset.dataset))
        print(NUM_TRAIN_SEED_ANNOTATIONS / 2 + NUM_VALID_SEED_ANNOTATIONS / 2)
        assert len(positive_dataset.dataset) >= NUM_TRAIN_SEED_ANNOTATIONS / 2 + NUM_VALID_SEED_ANNOTATIONS / 2
        assert len(negative_dataset.dataset) >= NUM_TRAIN_SEED_ANNOTATIONS / 2 + NUM_VALID_SEED_ANNOTATIONS / 2

        train_positive_dataset = deepcopy(positive_dataset)
        train_positive_dataset.dataset = train_positive_dataset.dataset.select(range(NUM_TRAIN_SEED_ANNOTATIONS // 2))
        train_negative_dataset = deepcopy(negative_dataset)
        train_negative_dataset.dataset = train_negative_dataset.dataset.select(range(NUM_TRAIN_SEED_ANNOTATIONS // 2))

        valid_positive_dataset = deepcopy(positive_dataset)
        valid_positive_dataset.dataset = valid_positive_dataset.dataset.select(range(NUM_TRAIN_SEED_ANNOTATIONS // 2, NUM_TRAIN_SEED_ANNOTATIONS // 2 + NUM_VALID_SEED_ANNOTATIONS // 2))
        valid_negative_dataset = deepcopy(negative_dataset)
        valid_negative_dataset.dataset = valid_negative_dataset.dataset.select(range(NUM_TRAIN_SEED_ANNOTATIONS // 2, NUM_TRAIN_SEED_ANNOTATIONS // 2 + NUM_VALID_SEED_ANNOTATIONS // 2))
        
        num_train_augmentations = 0
        train_datasets = []
        while num_train_augmentations < max(MAX_ENCODER_TRAIN_INPUTS, MAX_LLM_TRAIN_INPUTS):
            dataset = MultiLengthDifferenceRecognitionBenchmark(
                positive_dataset=train_positive_dataset,
                negative_dataset=train_negative_dataset,
                positive_ratio=POSITIVE_RATIO,
                max_sentences_per_document=MAX_SENTENCES_PER_DOCUMENT,
                max_inversions=MAX_INVERSIONS,
                seed=num_train_augmentations,
                pl=args.project_labels,
            ).to_dataset(both_directions=True)
            train_datasets.append(dataset)
            num_train_augmentations = sum(len(d) for d in train_datasets)
        train_dataset = concatenate_datasets(train_datasets)
        train_dataset = train_dataset.shuffle(seed=42)
        assert len(train_dataset) >= max(MAX_ENCODER_TRAIN_INPUTS, MAX_LLM_TRAIN_INPUTS)

        num_valid_augmentations = 0
        valid_datasets = []
        while num_valid_augmentations < max(MAX_ENCODER_VALID_INPUTS, MAX_LLM_VALID_INPUTS):
            dataset = MultiLengthDifferenceRecognitionBenchmark(
                positive_dataset=valid_positive_dataset,
                negative_dataset=valid_negative_dataset,
                positive_ratio=POSITIVE_RATIO,
                max_sentences_per_document=MAX_SENTENCES_PER_DOCUMENT,
                max_inversions=MAX_INVERSIONS,
                seed=num_valid_augmentations,
                pl=args.project_labels,
            ).to_dataset(both_directions=True)
            valid_datasets.append(dataset)
            num_valid_augmentations = sum(len(d) for d in valid_datasets)
        valid_dataset = concatenate_datasets(valid_datasets)
        valid_dataset = valid_dataset.shuffle(seed=42)
        assert len(valid_dataset) >= max(MAX_ENCODER_VALID_INPUTS, MAX_LLM_VALID_INPUTS)

    else:
        train_datasets = []
        valid_datasets = []

        for lang in ["de", "es", "fr", "ja", "ko", "zh", "en", "it"]:
            num_train_augmentations = 0
            num_valid_augmentations = 0

            print(lang)
            positive_dataset = ISTSDataset(split="train", tgt_lang=lang)
            positive_dataset.dataset = positive_dataset.dataset.shuffle(seed=42)
            
            negative_dataset = PAWSXDataset(split="train", language=lang)
            negative_dataset.dataset = negative_dataset.dataset.shuffle(seed=42)

            print(len(positive_dataset.dataset))
            print(len(negative_dataset.dataset))

            # change the 7s to 2s if working with multilingual-large
            assert len(positive_dataset.dataset) >= NUM_TRAIN_SEED_ANNOTATIONS / 7 + NUM_VALID_SEED_ANNOTATIONS / 7
            assert len(negative_dataset.dataset) >= NUM_TRAIN_SEED_ANNOTATIONS / 7 + NUM_VALID_SEED_ANNOTATIONS / 7

            train_positive_dataset = deepcopy(positive_dataset)
            train_positive_dataset.dataset = train_positive_dataset.dataset.select(range(NUM_TRAIN_SEED_ANNOTATIONS // 7))
            train_negative_dataset = deepcopy(negative_dataset)
            train_negative_dataset.dataset = train_negative_dataset.dataset.select(range(NUM_TRAIN_SEED_ANNOTATIONS // 7))

            valid_positive_dataset = deepcopy(positive_dataset)
            valid_positive_dataset.dataset = valid_positive_dataset.dataset.select(range(NUM_TRAIN_SEED_ANNOTATIONS // 7, NUM_TRAIN_SEED_ANNOTATIONS // 7 + NUM_VALID_SEED_ANNOTATIONS // 7))
            valid_negative_dataset = deepcopy(negative_dataset)
            valid_negative_dataset.dataset = valid_negative_dataset.dataset.select(range(NUM_TRAIN_SEED_ANNOTATIONS // 7, NUM_TRAIN_SEED_ANNOTATIONS // 7 + NUM_VALID_SEED_ANNOTATIONS // 7))

            while num_train_augmentations < max(MAX_ENCODER_TRAIN_INPUTS//7, MAX_LLM_TRAIN_INPUTS//7):
                dataset = MultiLengthDifferenceRecognitionBenchmark(
                    positive_dataset=train_positive_dataset,
                    negative_dataset=train_negative_dataset,
                    positive_ratio=POSITIVE_RATIO,
                    max_sentences_per_document=MAX_SENTENCES_PER_DOCUMENT,
                    max_inversions=MAX_INVERSIONS,
                    seed=num_train_augmentations,
                    pl=args.project_labels,
                ).to_dataset(both_directions=True)
                train_datasets.append(dataset)
                num_train_augmentations = sum(len(d) for d in train_datasets)

            while num_valid_augmentations < max(MAX_ENCODER_VALID_INPUTS//7, MAX_LLM_VALID_INPUTS//7):
                dataset = MultiLengthDifferenceRecognitionBenchmark(
                    positive_dataset=valid_positive_dataset,
                    negative_dataset=valid_negative_dataset,
                    positive_ratio=POSITIVE_RATIO,
                    max_sentences_per_document=MAX_SENTENCES_PER_DOCUMENT,
                    max_inversions=MAX_INVERSIONS,
                    seed=num_valid_augmentations,
                    pl=args.project_labels,
                ).to_dataset(both_directions=True)
                valid_datasets.append(dataset)
                num_valid_augmentations = sum(len(d) for d in valid_datasets)

        train_dataset = concatenate_datasets(train_datasets)
        train_dataset = train_dataset.shuffle(seed=42)
        assert len(train_dataset) >= max(MAX_ENCODER_TRAIN_INPUTS//7, MAX_LLM_TRAIN_INPUTS//7)
        print(len(train_dataset))
        valid_dataset = concatenate_datasets(valid_datasets)
        valid_dataset = valid_dataset.shuffle(seed=42)
        assert len(valid_dataset) >= max(MAX_ENCODER_VALID_INPUTS//7, MAX_LLM_VALID_INPUTS//7)
        print(len(valid_dataset))

   
    with jsonlines.open(train_encoder_path, "w") as f:
        samples = train_dataset.select(range(MAX_ENCODER_TRAIN_INPUTS))
        for sample in samples:
            sample = dict(sample)
            f.write(sample)

    with jsonlines.open(valid_encoder_path, "w") as f:
        samples = valid_dataset.select(range(MAX_ENCODER_VALID_INPUTS))
        for sample in samples:
            sample = dict(sample)
            f.write(sample)

    with jsonlines.open(train_llm_path, "w") as f:
        samples = train_dataset.select(range(MAX_LLM_TRAIN_INPUTS))
        for sample in samples:
            prompt = prompt_template.render(
                sentence1=json.dumps(sample["text_a"].split()),
                sentence2=json.dumps(sample["text_b"].split()),
            )
            gold_response = {
                "sentence1": [[token, LLM_LABELS_MAP[label]] for token, label in zip(sample["text_a"].split(), sample["labels_a"])],
                "sentence2": [[token, LLM_LABELS_MAP[label]] for token, label in zip(sample["text_b"].split(), sample["labels_b"])],
            }
            f.write({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(gold_response)},
            ]})

    with jsonlines.open(valid_llm_path, "w") as f:
        samples = valid_dataset.select(range(MAX_LLM_VALID_INPUTS))
        for sample in samples:
            prompt = prompt_template.render(
                sentence1=json.dumps(sample["text_a"].split()),
                sentence2=json.dumps(sample["text_b"].split()),
            )
            gold_response = {
                "sentence1": [[token, LLM_LABELS_MAP[label]] for token, label in zip(sample["text_a"].split(), sample["labels_a"])],
                "sentence2": [[token, LLM_LABELS_MAP[label]] for token, label in zip(sample["text_b"].split(), sample["labels_b"])],
            }
            f.write({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(gold_response)},
            ]})
