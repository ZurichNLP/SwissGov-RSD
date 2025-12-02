from fireworks import LLM
from pathlib import Path
from typing import Union
import os
import json
import uuid
from tqdm import tqdm
import jsonlines
from datetime import datetime
import argparse as ap


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--lang', type=str, default=None)
    args = parser.parse_args()
    # Load dataset
    if args.lang:
        test_data_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'llm_inputs' / f'test_admin_{args.lang}.jsonl'
    else:
        test_data_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'llm_inputs' / 'test.jsonl'
    assert test_data_path.exists(), f"Test data not found at {test_data_path}"

    start = 0
    end = 1
    i = None
    if args.lang:
        outfile = Path(__file__).parent.parent / 'data' / 'evaluation' / 'llm_predictions' / f'llama-405b-{start}-{end}-admin-{args.lang}.jsonl'
    else:
        outfile = Path(__file__).parent.parent / 'data' / 'evaluation' / 'llm_predictions' / f'llama-405b-{start}-{end}.jsonl'

    api_key = Path(__file__).parent.parent / '.openai' / 'fireworks.txt'

    # Initialize Fireworks LLM
    llm = LLM(
        model="accounts/fireworks/models/llama-v3p1-405b-instruct",
        api_key=open(api_key).read().strip(),
        deployment_type="serverless"
    )

    # Read the JSONL file directly
    predictions = []
    with open(test_data_path, 'r') as f, open(outfile, 'a') as f_out:

        for i, line in enumerate(tqdm(f, desc=f"Processing items {start}-{end} (current: {i if i else start})", initial=start)):
            if i < start:
                predictions.append(json.loads(line))
                continue
            if i >= end:
                break
            item = json.loads(line)
            
            # Extract sentences from the message content
            content = item["messages"][0]["content"]
            # Find the last occurrence of "Sentence 1:" and "Sentence 2:"
            sentence1_start = content.rfind("Sentence 1: [") + len("Sentence 1: [")
            sentence1_end = content.find("]", sentence1_start)
            sentence2_start = content.rfind("Sentence 2: [") + len("Sentence 2: [")
            sentence2_end = content.find("]", sentence2_start)
            
            sentence1 = content[sentence1_start:sentence1_end].replace('"', '').split(", ")
            sentence2 = content[sentence2_start:sentence2_end].replace('"', '').split(", ")
        
            # Use Fireworks API
            response = llm.chat.completions.create(
                messages=item["messages"],
                max_tokens=65536,
                top_p=1,
                top_k=0,
                presence_penalty=0,
                frequency_penalty=0,
                temperature=0,
                stream=True
            )
            
            output = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    output += chunk.choices[0].delta.content
            
            # Create API request structure
            api_request = {
                "custom_id": f"request-{len(predictions)}",
                "method": "POST",
                "url": "accounts/fireworks/models/llama-v3p1-405b-instruct",
                "body": {
                    "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
                    "messages": item["messages"],
                    "max_tokens": 65536,
                    "top_p": 1,
                    "top_k": 0,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "temperature": 0,
                    "stream": True
                }
            }

            # Create API response structure
            api_response = {
                "id": f"ists_{len(predictions)}",
                "custom_id": api_request["custom_id"],
                "response": {
                    "status_code": 200,
                    "request_id": uuid.uuid4().hex,
                    "body": {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                        "object": "chat.completion",
                        "created": int(datetime.now().timestamp()),
                        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": output,
                                "refusal": None
                            },
                            "logprobs": None,
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                            "prompt_tokens_details": {
                                "cached_tokens": 0,
                                "audio_tokens": 0
                            },
                            "completion_tokens_details": {
                                "reasoning_tokens": 0,
                                "audio_tokens": 0,
                                "accepted_prediction_tokens": 0,
                                "rejected_prediction_tokens": 0
                            }
                        },
                        "service_tier": "default",
                        "system_fingerprint": f"fp_{uuid.uuid4().hex[:12]}"
                    }
                },
                "error": None,
                "provider": "fireworks"
            }

            pred = {
                "item_id": f"admin_{args.lang}_{len(predictions)}",
                "sentence1": " ".join(sentence1),
                "sentence2": " ".join(sentence2),
                "api_request": api_request,
                "api_response": api_response,
                "provider": "fireworks"
            }
            predictions.append(pred)
            f_out.write(json.dumps(pred, ensure_ascii=False) + "\n")
            f_out.flush()

if __name__ == "__main__":
    main()