import openai
import json
import jinja2
from pathlib import Path
import argparse as ap
import os
import sys
import jsonlines
from openai import OpenAI
from tqdm import tqdm

# Get the absolute path to the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the root dir to sys.path so Python can find the 'rsd' package
sys.path.insert(0, ROOT_DIR)

from rsd.data.ists import ISTSDataset, ISTSSample
from rsd.data.pawsx import PAWSXDataset, PAWSXSample


LLM_LABELS_MAP = {
    0.0: 5,
    0.2: 4,
    0.4: 3,
    0.6: 2,
    0.8: 1,
    1.0: 0,
    -1.0: -1,
}

REVERSE_LLM_LABELS_MAP = {v: k for k, v in LLM_LABELS_MAP.items()}
CLIENT = OpenAI(api_key=open('.openai/key.txt', 'r', encoding='utf-8').read().strip())

# load english data
ists_en = ISTSDataset(split="train", tgt_lang="en")
pawsx_en = PAWSXDataset(split="train", language="en")

# Create lookup dictionaries for faster matching
ISTS_LOOKUP = {}
PAWSX_LOOKUP = {}

for ists_sample in ists_en.get_samples():   
    ISTS_LOOKUP[" ".join(ists_sample.tokens_a)] = ists_sample
    ISTS_LOOKUP[" ".join(ists_sample.tokens_b)] = ists_sample

for pawsx_sample in pawsx_en.get_samples():
    PAWSX_LOOKUP[" ".join(pawsx_sample.tokens_a)] = pawsx_sample
    PAWSX_LOOKUP[" ".join(pawsx_sample.tokens_b)] = pawsx_sample


def keep_one_side(data_to_annotate, existing_projections, outpath):
    print("Keeping one side")
    with jsonlines.open(outpath, 'w') as f:
        for output, sample in tqdm(zip(list(existing_projections), list(data_to_annotate)), desc="Processing existing projections"):
            print(output)
            print(sample)
            faulty_dict = {}
            try:
                print(sample["text_a"], len(output["text_a"]))
                print(sample["text_a"], len(sample["text_a"]))
                print(output["labels_a"], len(output["labels_a"]))
                print(sample["labels_a"], len(sample["labels_a"]))
                assert output["text_a"] == sample["text_a"], "Text a does not match"
            except:
                if "text_a" not in faulty_dict:
                    faulty_dict["text_a"] = 1
                else:
                    faulty_dict["text_a"] += 1
                output["text_a"] = sample["text_a"]

            try:
                print(output["text_b"], len(output["text_b"]))
                print(sample["text_b"], len(sample["text_b"]))
                print(output["labels_b"], len(output["labels_b"]))
                print(sample["labels_b"], len(sample["labels_b"]))
                assert output["text_b"] == sample["text_b"], "Text b does not match"
            except:
                if "text_b" not in faulty_dict:
                    faulty_dict["text_b"] = 1
                else:
                    faulty_dict["text_b"] += 1
                output["text_b"] = sample["text_b"]
                
            if not all(label == -1.0 for label in sample["labels_b"]):
                try:
                    assert output["labels_b"] == sample["labels_b"], "Labels b do not match"
                except:
                    if "labels_b" not in faulty_dict:
                        faulty_dict["labels_b"] = 1
                    else:
                        faulty_dict["labels_b"] += 1
                    output["labels_b"] = sample["labels_b"]
            
            if not all(label == -1.0 for label in sample["labels_a"]):
                try:
                    assert output["labels_a"] == sample["labels_a"], "Labels a do not match"
                except:
                    if "labels_a" not in faulty_dict:
                        faulty_dict["labels_a"] = 1
                    else:
                        faulty_dict["labels_a"] += 1
                    output["labels_a"] = sample["labels_a"]
            
            if len(output['text_a'].split()) != len(output['labels_a']):
                print("Correcting labels_a length")
                target_len = len(output['text_a'].split())
                if len(output['labels_a']) < target_len:
                    # Pad with -1.0 if labels are too short
                    output['labels_a'].extend([-1.0] * (target_len - len(output['labels_a'])))
                else:
                    # Truncate if labels are too long
                    output['labels_a'] = output['labels_a'][:target_len]

            if len(output['text_b'].split()) != len(output['labels_b']):
                print("Correcting labels_b length")
                target_len = len(output['text_b'].split())
                if len(output['labels_b']) < target_len:
                    # Pad with -1.0 if labels are too short 
                    output['labels_b'].extend([-1.0] * (target_len - len(output['labels_b'])))
                else:
                    # Truncate if labels are too long
                    output['labels_b'] = output['labels_b'][:target_len]

            print(output['text_a'].split())
            print(output['labels_a'])
            print(len(output['text_a'].split()))
            print(len(output['labels_a']))
            assert len(output['text_a'].split()) == len(output['labels_a']), "Text a and labels a do not have the same length"
            assert len(output['text_b'].split()) == len(output['labels_b']), "Text b and labels b do not have the same length"

            print(output)
            f.write(output)


def project_labels(sample, lang):
    projection_prompt_path = Path(__file__).parent.parent / "prompt_templates" / f"label_projection_{lang}.txt"
    projection_prompt = open(projection_prompt_path, 'r', encoding='utf-8').read()

    prompt_template_path = Path(__file__).parent.parent / "prompt_templates" / f"projection_template.txt"
    prompt_template = jinja2.Template(open(prompt_template_path, 'r', encoding='utf-8').read())
    
    if not __name__ == "__main__":
        sample = {
            "text_a": " ".join(sample.tokens_a),
            "text_b": " ".join(sample.tokens_b),
            "labels_a": sample.labels_a,
            "labels_b": sample.labels_b,
        }

    # Try to find match in ISTS dataset if the script was run as main script
    matched_sample = (ISTS_LOOKUP.get(sample["text_a"]) or 
                    ISTS_LOOKUP.get(sample["text_b"]) or
                    PAWSX_LOOKUP.get(sample["text_a"]) or 
                    PAWSX_LOOKUP.get(sample["text_b"]))

    assert matched_sample is not None, "No matching English sample found"

    sentence1_en = matched_sample.tokens_a
    sentence2_en = matched_sample.tokens_b
    labels1_en = matched_sample.labels_a
    labels2_en = matched_sample.labels_b

    assert len(sentence1_en) == len(labels1_en), "Sentence 1 and labels 1 do not have the same length"
    assert len(sentence2_en) == len(labels2_en), "Sentence 2 and labels 2 do not have the same length"

    # convert labels according to LLM_LABELS_MAP
    labels1_en = [LLM_LABELS_MAP[label] for label in labels1_en]
    labels2_en = [LLM_LABELS_MAP[label] for label in labels2_en]

    # Define API call
    response = CLIENT.responses.create(
        model="gpt-4o-mini-2024-07-18",
        input=[
            {"role": "system", "content": projection_prompt},
            {"role": "user", "content": prompt_template.render(
                sentence1_en=sentence1_en,
                sentence2_en=sentence2_en,
                labels1_en=labels1_en,
                labels2_en=labels2_en,
                text_a=sample["text_a"],
                text_b=sample["text_b"]
            )}
        ],
        temperature=0.0,
        text={
            "format": {
                "type": "json_schema",
                "name": "projected_labels",
                "schema": {
                    "type": "object",
                    "properties": {
                        "text_a": {"type": "string"},
                        "text_b": {"type": "string"},
                        "labels_a": {"type": "array", "items": {"type": "number"}},
                        "labels_b": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["text_a", "text_b", "labels_a", "labels_b"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        }
    )


    output = json.loads(response.output_text)

    # convert labels according to LLM_LABELS_MAP
    output["labels_a"] = [REVERSE_LLM_LABELS_MAP[label] for label in output["labels_a"]]
    output["labels_b"] = [REVERSE_LLM_LABELS_MAP[label] for label in output["labels_b"]]

    faulty_dict = {}

    try:
        print(output["text_a"], len(output["text_a"]))
        print(sample["text_a"], len(sample["text_a"]))
        print(output["labels_a"], len(output["labels_a"]))
        print(sample["labels_a"], len(sample["labels_a"]))
        assert output["text_a"] == sample["text_a"], "Text a does not match"
    except:
        if "text_a" not in faulty_dict:
            faulty_dict["text_a"] = 1
        else:
            faulty_dict["text_a"] += 1
        output["text_a"] = sample["text_a"]

    try:
        print(output["text_b"], len(output["text_b"]))
        print(sample["text_b"], len(sample["text_b"]))
        print(output["labels_b"], len(output["labels_b"]))
        print(sample["labels_b"], len(sample["labels_b"]))
        assert output["text_b"] == sample["text_b"], "Text b does not match"
    except:
        if "text_b" not in faulty_dict:
            faulty_dict["text_b"] = 1
        else:
            faulty_dict["text_b"] += 1
        output["text_b"] = sample["text_b"]
        
    if not all(label == -1.0 for label in sample["labels_b"]):
        try:
            assert output["labels_b"] == sample["labels_b"], "Labels b do not match"
        except:
            if "labels_b" not in faulty_dict:
                faulty_dict["labels_b"] = 1
            else:
                faulty_dict["labels_b"] += 1
            output["labels_b"] = sample["labels_b"]
    
    if not all(label == -1.0 for label in sample["labels_a"]):
        try:
            assert output["labels_a"] == sample["labels_a"], "Labels a do not match"
        except:
            if "labels_a" not in faulty_dict:
                faulty_dict["labels_a"] = 1
            else:
                faulty_dict["labels_a"] += 1
            output["labels_a"] = sample["labels_a"]
    
    if len(output['text_a'].split()) != len(output['labels_a']):
        print("Correcting labels_a length")
        target_len = len(output['text_a'].split())
        if len(output['labels_a']) < target_len:
            # Pad with -1.0 if labels are too short
            output['labels_a'].extend([-1.0] * (target_len - len(output['labels_a'])))
        else:
            # Truncate if labels are too long
            output['labels_a'] = output['labels_a'][:target_len]

    if len(output['text_b'].split()) != len(output['labels_b']):
        print("Correcting labels_b length")
        target_len = len(output['text_b'].split())
        if len(output['labels_b']) < target_len:
            # Pad with -1.0 if labels are too short 
            output['labels_b'].extend([-1.0] * (target_len - len(output['labels_b'])))
        else:
            # Truncate if labels are too long
            output['labels_b'] = output['labels_b'][:target_len]

    print(output['text_a'].split())
    print(output['labels_a'])
    print(len(output['text_a'].split()))
    print(len(output['labels_a']))
    assert len(output['text_a'].split()) == len(output['labels_a']), "Text a and labels a do not have the same length"
    assert len(output['text_b'].split()) == len(output['labels_b']), "Text b and labels b do not have the same length"

    if __name__ == "__main__":
        return output
    else:
        return ISTSSample(
            tokens_a=output["text_a"].split(),
            tokens_b=output["text_b"].split(),
            labels_a=output["labels_a"],
            labels_b=output["labels_b"],
        )



def main():
    parser = ap.ArgumentParser()
    parser.add_argument("path_to_data_to_annotate", type=str)
    parser.add_argument("--lang", type=str, help="Language to use for training data (e.g.: de, es, multi)")
    parser.add_argument("--keep_one_side", action="store_true")
    parser.add_argument("--api_key", type=str, help="API key for OpenAI")
    args = parser.parse_args()


    # load data to annotate
    data_to_annotate = jsonlines.open(args.path_to_data_to_annotate, 'r')

    
    if args.keep_one_side:
        outpath = args.path_to_data_to_annotate.replace(".jsonl", "_projected_asserted.jsonl")
    else:
        outpath = args.path_to_data_to_annotate.replace(".jsonl", "_projected_asserted.jsonl")

    if os.path.exists(outpath):
        print(f"Output file {outpath} already exists.")

        # create look up for already projected samples
        already_projected_pairs = set()
        for sample in jsonlines.open(outpath, 'r'):
            already_projected_pairs.add((sample["text_a"], sample["text_b"]))

        writing_mode = "a"
    else:
        writing_mode = "w"

    counter = 0

    if args.keep_one_side:
        existing_projections = jsonlines.open(args.path_to_data_to_annotate.replace(".jsonl", "_projected.jsonl"), 'r')
        keep_one_side(data_to_annotate, existing_projections, outpath)

    else:
        # Process each sample
        with jsonlines.open(outpath, writing_mode) as f:

            counter = 0
            data_to_annotate = list(data_to_annotate)
            for sample in data_to_annotate[8498:]:
                #counter += 1
                #if counter > 10:
                    #break
                # skip either text_a or text_b if already in output file f
                if writing_mode == "a":
                    counter += 1
                    """if (sample["text_a"], sample["text_b"]) in already_projected_pairs:
                        print(f"{counter} already projected")
                        continue"""
                output = project_labels(sample, args.lang)
                f.write(output)



if __name__ == "__main__":
    main()