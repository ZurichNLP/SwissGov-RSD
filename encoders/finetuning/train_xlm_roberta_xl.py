import argparse
from dataclasses import dataclass

from accelerate import load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    EarlyStoppingCallback,
    AutoConfig,
)

from encoders.data_collator import DataCollatorForTokenRegression
from encoders.modeling_xlm_roberta_xl import XLMRobertaXLForTokenRegression
from encoders.finetuning.data import load_encoder_training_data


@dataclass
class ModelConfig:
    model_name_or_path: str
    model_revision: str = "main"
    trust_remote_code: bool = False
    torch_dtype: str = "auto"


def main(training_args, model_args):
    ################
    # Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token_id is not None:
            tokenizer.pad_token = tokenizer.sep_token

    ################
    # Dataset
    ################
    dataset = load_encoder_training_data(tokenizer)
    collator = DataCollatorForTokenRegression(tokenizer=tokenizer)

    ################
    # Training
    ################
    weights_location = hf_hub_download(model_args.model_name_or_path, "pytorch_model.bin")
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=2)  # Weight loading not working with num_labels=1
    model = XLMRobertaXLForTokenRegression(config, torch_dtype=model_args.torch_dtype)
    model = load_checkpoint_and_dispatch(
        model, weights_location, device_map="auto", no_split_module_classes=["XLMRobertaXLEmbeddings", "XLMRobertaXLLayer"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (TrainingArguments, ModelConfig)
    parser = argparse.ArgumentParser()
    
    # Add arguments for TrainingArguments
    training_group = parser.add_argument_group("Training Arguments")
    for field in TrainingArguments.__dataclass_fields__.values():
        training_group.add_argument(f"--{field.name}", type=field.type, default=field.default)
    
    # Add arguments for ModelConfig
    model_group = parser.add_argument_group("Model Arguments")
    for field in ModelConfig.__dataclass_fields__.values():
        model_group.add_argument(f"--{field.name}", type=field.type, default=field.default)
    
    return parser


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, ModelConfig))
    args = parser.parse_args()
    
    # Split args into training_args and model_args
    training_dict = {k: v for k, v in vars(args).items() if k in TrainingArguments.__dataclass_fields__}
    model_dict = {k: v for k, v in vars(args).items() if k in ModelConfig.__dataclass_fields__}
    
    training_args = TrainingArguments(**training_dict)
    model_args = ModelConfig(**model_dict)
    
    main(training_args, model_args)
