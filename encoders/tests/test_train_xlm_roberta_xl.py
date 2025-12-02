import unittest
from pathlib import Path

from transformers import TrainingArguments

from encoders.finetuning.train_xlm_roberta_xl import ModelConfig, main


class TestBertTraining(unittest.TestCase):

    def setUp(self):
        self.model_name = "hf-tiny-model-private/tiny-random-XLMRobertaXLForTokenClassification"
        self.model_path = Path(__file__).parent / "temp_xlm_model"

    def test_training_bert(self):
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=str(self.model_path),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            max_steps=1,
            load_best_model_at_end=True,
            save_strategy="epoch",
            eval_strategy="epoch",
        )
        
        model_args = ModelConfig(
            model_name_or_path=self.model_name,
            trust_remote_code=True,
        )

        # Run training
        main(training_args, model_args)
        
        # Verify training artifacts exist
        self.assertTrue(Path(training_args.output_dir).exists())
        self.assertTrue((Path(training_args.output_dir) / "model.safetensors").exists())

    def tearDown(self):
        # Cleanup temp files and model
        if self.model_path.exists():
            import shutil
            shutil.rmtree(self.model_path)


if __name__ == '__main__':
    unittest.main()
