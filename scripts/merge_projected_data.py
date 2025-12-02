import json
import jsonlines
import random


def main():
    for split in ["train", "valid"]:
        path_de = f"data/finetuning/encoder_inputs/{split}_de_projected.jsonl"
        path_it = f"data/finetuning/encoder_inputs/{split}_it_projected.jsonl"
        path_fr = f"data/finetuning/encoder_inputs/{split}_fr_projected.jsonl"

        samples = []
        with open(path_de, "r") as f_de, open(path_it, "r") as f_it, open(path_fr, "r") as f_fr:
            for line_de, line_it, line_fr in zip(f_de, f_it, f_fr):
                item_de = json.loads(line_de)
                item_it = json.loads(line_it)
                item_fr = json.loads(line_fr)

                samples.append(item_de)
                samples.append(item_it)
                samples.append(item_fr)
            
            random.shuffle(samples, random.seed(42))

            if split == "train":
                assert len(samples) == 30000, f"Expected 30000 samples, got {len(samples)}"
            else:
                assert len(samples) == 600, f"Expected 600 samples, got {len(samples)}"

        with jsonlines.open(f"data/finetuning/encoder_inputs/{split}_de_it_fr_projected.jsonl", "w") as f_out:
            for sample in samples:
                f_out.write(sample)

if __name__ == "__main__":
    main()