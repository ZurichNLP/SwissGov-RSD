import timeit
from pathlib import Path

from transformers import pipeline

from encoders.encoder_recognizer import EncoderDifferenceRecognizer
from rsd.recognizers import DiffAlign

sentence1_short = "The quick brown fox jumps over the lazy dog ."
sentence2_short = "A rapid blue dog is jumping over a lazy mouse ."

sentence1_long = " ".join(5 * [sentence1_short])
sentence2_long = " ".join(5 * [sentence2_short])

# Fine-tuned
print("Fine-tuned")
encoder_recognizer = EncoderDifferenceRecognizer("jvamvas/xlm-roberta-xl-rsd")

def run_finetuned_short():
    encoder_recognizer.predict(sentence1_short, sentence2_short)

def run_finetuned_long():
    encoder_recognizer.predict(sentence1_long, sentence2_long)

short_time = timeit.timeit(run_finetuned_short, number=100) / 100
print(f"Average latency for short sentences: {short_time:.3f} seconds")

long_time = timeit.timeit(run_finetuned_long, number=100) / 100
print(f"Average latency for long sentences: {long_time:.3f} seconds")

del encoder_recognizer


# DiffAlign
print("DiffAlign")
diffalign = DiffAlign("facebook/xlm-roberta-xl")

def run_diffalign_short():
    diffalign.predict(sentence1_short, sentence2_short)

def run_diffalign_long():
    diffalign.predict(sentence1_long, sentence2_long)

short_time = timeit.timeit(run_diffalign_short, number=100) / 100
print(f"Average latency for short sentences: {short_time:.3f} seconds")

long_time = timeit.timeit(run_diffalign_long, number=100) / 100
print(f"Average latency for long sentences: {long_time:.3f} seconds")

del diffalign


# LLM
print("LLM")
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",
)

template_path = Path(__file__).parent.parent / "prompt_templates" / "template.txt"
prompt_template = template_path.read_text()

def run_llm_short():
    chat = [{"role": "user", "content": prompt_template.replace("{{ sentence1 }}", "[" + ", ".join([f'"{word}"' for word in sentence1_short.split()]) + "]").replace("{{ sentence2 }}", "[" + ", ".join([f'"{word}"' for word in sentence2_short.split()]) + "]")}]
    print(chat)
    prompt = pipe.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    output = pipe(
        prompt,
        max_new_tokens=4096,
        do_sample=False,
        temperature=None,
        top_p=None,
    )[0]["generated_text"]

def run_llm_long():
    chat = [{"role": "user", "content": prompt_template.replace("{{ sentence1 }}", "[" + ", ".join([f'"{word}"' for word in sentence1_long.split()]) + "]").replace("{{ sentence2 }}", "[" + ", ".join([f'"{word}"' for word in sentence2_long.split()]) + "]")}]
    print(chat)
    prompt = pipe.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    output = pipe(
        prompt,
        max_new_tokens=4096,
        do_sample=False,
        temperature=None,
        top_p=None,
    )[0]["generated_text"]

short_time = timeit.timeit(run_llm_short, number=10)
print(f"Average latency for short sentences: {short_time:.3f} seconds")

long_time = timeit.timeit(run_llm_long, number=10)
print(f"Average latency for long sentences: {long_time:.3f} seconds")
