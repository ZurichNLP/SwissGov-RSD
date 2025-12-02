import unittest
from pathlib import Path

from llama.train_llama import main, ScriptArguments, SFTConfig, ModelConfig


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.model_name = "yujiepan/llama-3.3-tiny-random"
        self.model_path = Path(__file__).parent / "temp_llama_model"

    def test_training_llama_toy(self):
        script_args = ScriptArguments(
            dataset_name="custom",
            dataset_train_split="train",
            dataset_test_split="valid",
        )
        
        training_args = SFTConfig(
            output_dir=str(self.model_path),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=1,  # Only train for 1 step
            learning_rate=2e-4,
            logging_steps=1,
            eval_strategy="steps",
            gradient_checkpointing=True,
            load_best_model_at_end=True,
            max_seq_length=4096,
        )
        
        model_args = ModelConfig(
            model_name_or_path=self.model_name,
            use_peft=True,
            lora_r=32,
            lora_alpha=16,
        )

        # Run training
        main(script_args, training_args, model_args)
        
        # Verify training artifacts exist
        self.assertTrue(Path(training_args.output_dir).exists())
        self.assertTrue((Path(training_args.output_dir) / "checkpoint-1" / "adapter_model.safetensors").exists())

    def tearDown(self):
        # Cleanup temp files and model
        if self.model_path.exists():
            import shutil
            shutil.rmtree(self.model_path)

if __name__ == '__main__':
    unittest.main()
