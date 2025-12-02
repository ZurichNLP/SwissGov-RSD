import json
from pathlib import Path
import sys
import os
import argparse as ap

import jsonlines
import jinja2

parser = ap.ArgumentParser()
parser.add_argument("--lang", type=str, help="Language to use for test data")
parser.add_argument("--sents", action="store_true", help="If used, this will be sentence-level data only. Does include negative sentence-level data from PAWSX.")
args = parser.parse_args()

# Get the absolute path to the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the root dir to sys.path so Python can find the 'rsd' package
sys.path.insert(0, ROOT_DIR)

from rsd.experiments.utils import load_summary_benchmarks

SUBSET_NAMES = [
    "ists",
    "ists_negatives",
    "ists_documents",
    "ists_permutations",
    "ists_de",
    "ists_es",
    "ists_fr",
    "ists_ja",
    "ists_ko",
    "ists_zh",
    "ists_it"
]

NUM_TESTS_SAMPLES_PER_SUBSET = 100
TEST_SUBSETS = [
    "ists",
    "ists_negatives",
    "ists_documents",
    "ists_permutations",
    "ists_de",
    "ists_es",
    "ists_fr",
    "ists_ja",
    "ists_ko",
    "ists_zh",
    "ists_it"
]

LLM_LABELS_MAP = {
    0.0: 5,
    0.2: 4,
    0.4: 3,
    0.6: 2,
    0.8: 1,
    1.0: 0,
    -1.0: -1,
}

if args.lang:
    prompt_template_path = Path(__file__).parent.parent / "prompt_templates" / f"template_{args.lang}.txt"
else:
    prompt_template_path = Path(__file__).parent.parent / "prompt_templates" / "template.txt"
prompt_template = jinja2.Template(prompt_template_path.read_text())

data_dir = Path(__file__).parent.parent / "data"
if args.lang:
    test_gold_path = data_dir / "evaluation" / "gold_labels" / f"gold_{args.lang}.jsonl"
    test_llm_path = data_dir / "evaluation" / "llm_inputs" / f"test_{args.lang}.jsonl"
elif args.sents:
    test_gold_path = data_dir / "evaluation" / "gold_labels" / f"gold_{args.lang}_sents.jsonl"
    test_llm_path = data_dir / "evaluation" / "llm_inputs" / f"test_{args.lang}_sents.jsonl"
else:
    test_gold_path = data_dir / "evaluation" / "gold_labels" / "gold.jsonl"
    test_llm_path = data_dir / "evaluation" / "llm_inputs" / "test.jsonl"

test_subsets = load_summary_benchmarks("test", tgt_lang=args.lang if args.lang else "en")
test_subsets = {SUBSET_NAMES[i]: subset for i, subset in enumerate(test_subsets) if SUBSET_NAMES[i] in TEST_SUBSETS}
assert len(test_subsets) == len(TEST_SUBSETS)

with jsonlines.open(test_gold_path, "w") as f:
    for subset_name, subset in test_subsets.items():
        dataset = subset.to_dataset()
        assert len(dataset) >= NUM_TESTS_SAMPLES_PER_SUBSET
        dataset.shuffle(seed=42)
        dataset = dataset.map(lambda x: {"subset": subset_name})
        dataset = dataset.map(lambda x, idx: {"id": f"{subset_name}_{idx}"}, with_indices=True)
        samples = dataset.select(range(NUM_TESTS_SAMPLES_PER_SUBSET))
        for i, sample in enumerate(samples):
            f.write(sample)

with jsonlines.open(test_llm_path, "w") as f:
    for subset_name, subset in test_subsets.items():
        dataset = subset.to_dataset()
        assert len(dataset) >= NUM_TESTS_SAMPLES_PER_SUBSET
        dataset.shuffle(seed=42)
        dataset = dataset.map(lambda x: {"subset": subset_name})
        dataset = dataset.map(lambda x, idx: {"id": f"{subset_name}_{idx}"}, with_indices=True)
        samples = dataset.select(range(NUM_TESTS_SAMPLES_PER_SUBSET))
        for i, sample in enumerate(samples):
            prompt = prompt_template.render(
                sentence1=json.dumps(sample["text_a"].split()),
                sentence2=json.dumps(sample["text_b"].split()),
            )
            gold_response = {
                "sentence1": [[token, LLM_LABELS_MAP[round(label, 1)]] for token, label in zip(sample["text_a"].split(), sample["labels_a"])],
                "sentence2": [[token, LLM_LABELS_MAP[round(label, 1)]] for token, label in zip(sample["text_b"].split(), sample["labels_b"])],
            }
            f.write({
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "id": f"{subset_name}_{i}",
            })
