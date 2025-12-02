from pathlib import Path
from typing import Union
import os
import json
import uuid
from datetime import datetime

import jsonlines
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline


def main(
        base_model_name_or_path: Union[Path, str],
        adapter_model_name_or_path: Union[Path, str] = None,
):
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=base_model_name_or_path,
        device_map="auto",
        token=open(".openai/hf_key.txt").read(),  # Add token to pipeline
    )
    if adapter_model_name_or_path is not None:
        pipe.model.load_adapter(adapter_model_name_or_path)

    # Load dataset
    test_data_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'llm_inputs' / 'test_admin_de.jsonl'
    suffix = test_data_path.name.split('_')[-3:]
    assert test_data_path.exists(), f"Test data not found at {test_data_path}"
    
    # Read the JSONL file directly
    predictions = []
    with open(test_data_path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            
            # Extract sentences from the messages
            messages = item["messages"]
            user_message = messages[0]["content"]
            
            # Parse the input sentences from the message
            # The message contains the full prompt, we need to extract just the input sentences
            input_sentences = user_message.split("## Input to Annotate\n\n")[-1].strip()
            sentence1 = input_sentences.split("Sentence 1: ")[1].split("\n")[0]
            sentence2 = input_sentences.split("Sentence 2: ")[1].strip()
            
            # Generate output
            chat = [{"role": "user", "content": user_message}]
            prompt = pipe.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            
            output = pipe(
                prompt,
                max_new_tokens=4096,
                do_sample=False,
                temperature=None,
                top_p=None,
            )[0]["generated_text"]
            output = output[len(prompt):].strip()  # Remove prompt from output

            # Create API request structure
            api_request = {
                "custom_id": f"request-{len(predictions)}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": str(base_model_name_or_path),
                    "messages": [{"role": "user", "content": [{"type": "text", "text": user_message}]}],
                    "response_format": {"type": "json_object"}
                }
            }

            # Create API response structure
            api_response = {
                "id": f"batch_req_{uuid.uuid4().hex}",
                "custom_id": api_request["custom_id"],
                "response": {
                    "status_code": 200,
                    "request_id": uuid.uuid4().hex,
                    "body": {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                        "object": "chat.completion",
                        "created": int(datetime.now().timestamp()),
                        "model": str(base_model_name_or_path),
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
                            "prompt_tokens": len(prompt.split()),
                            "completion_tokens": len(output.split()),
                            "total_tokens": len(prompt.split()) + len(output.split()),
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
                "provider": "openai"
            }

            pred = {
                "item_id": f"ists_{len(predictions)}",
                "sentence1": sentence1,
                "sentence2": sentence2,
                "api_request": api_request,
                "api_response": api_response
            }
            predictions.append(pred)

    # Save predictions in out directory
    out_dir = Path(__file__).parent.parent / 'data' / 'evaluation' / 'llm_predictions'
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / (f"{str(base_model_name_or_path).replace('/', '_')}-{" ".join(suffix)}"
                             f"{'_' + str(adapter_model_name_or_path).replace('/', '_') if adapter_model_name_or_path else ''}.jsonl")
    with jsonlines.open(output_file, "w") as f:
        for pred in predictions:
            f.write(pred)

    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_model', type=str)
    parser.add_argument('--adapter', type=str, default=None)

    args = parser.parse_args()
    main(args.base_model, args.adapter)
