from openai import OpenAI
import json
import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06") #gpt-4o-2024-08-06; gpt-4o-mini-2024-07-18; o3-mini-2025-01-31; ft:gpt-4o-mini-2024-07-18:cl-uzh:rsd:Br8lkNWs
parser.add_argument("--lang", type=str, default=None) #en; de
#parser.add_argument("--output", type=str, default="data/evaluation/llm_predictions/test_output.jsonl")
args = parser.parse_args()

client = OpenAI(api_key=open(".openai/key.txt").read().strip())

if args.lang is None:
    output = f"data/evaluation/llm_predictions/{args.model}.jsonl"
    test_file = f"data/evaluation/llm_inputs/test.jsonl"
else:
    output = f"data/evaluation/llm_predictions/{args.model}_admin_{args.lang}.jsonl"
    test_file = f"data/evaluation/llm_inputs/test_admin_{args.lang}.jsonl"

start = 79
end = 235

with open(test_file, "r") as f_in, open(output, "a") as f_out:
    lines = f_in.readlines()
    for line in tqdm(lines[start:end]):
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
        
        # Create API request object
        api_request = {
            "custom_id": f"request-{item['id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.model,
                "messages": item["messages"],
                "response_format": {"type": "json_object"}
            }
        }
        
        # Make API call
        if "o3" in args.model:
            response = client.chat.completions.create(
                model=args.model,
                messages=item["messages"],
                response_format={"type": "json_object"},
                reasoning_effort="low"
            )

        else:
            response = client.chat.completions.create(
                model=args.model,
                messages=item["messages"],
                response_format={"type": "json_object"}
            )
        
        # Create API response object
        api_response = {
            "id": response.id,
            "custom_id": api_request["custom_id"],
            "response": {
                "status_code": 200,
                "request_id": response.id,
                "body": {
                    "id": response.id,
                    "object": "chat.completion",
                    "created": response.created,
                    "model": response.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content,
                            "refusal": None
                        },
                        "logprobs": None,
                        "finish_reason": response.choices[0].finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
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
                    "system_fingerprint": "fp_72ed7ab54c"
                }
            },
            "error": None,
        }
        
        # Create output object
        output = {
            "item_id": item["id"],
            "sentence1": " ".join(sentence1),
            "sentence2": " ".join(sentence2),
            "api_request": api_request,
            "api_response": api_response,
            "provider": "openai"
        }
        
        # Write to JSONL file with ensure_ascii=False to preserve special characters
        f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
        f_out.flush()

print(f"Results saved to {output}")