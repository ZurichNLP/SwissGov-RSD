import unittest
import torch
from transformers import AutoTokenizer
from encoders.data_collator import DataCollatorForTokenRegression


class TestDataCollatorForTokenRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cls.data_collator = DataCollatorForTokenRegression(
            tokenizer=cls.tokenizer,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            label_pad_token_id=-100
        )

    def test_basic_collation(self):
        features = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [0.1, 0.2, 0.3]
            },
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "labels": [0.4, 0.5]
            }
        ]

        batch = self.data_collator(features)

        # Check that batch contains expected keys
        self.assertTrue(all(key in batch for key in ["input_ids", "attention_mask", "labels"]))
        
        # Check tensor shapes and types
        self.assertEqual(batch["labels"].dtype, torch.float32)
        self.assertEqual(batch["labels"].shape, (2, 3))
        
        # Check padding values
        self.assertEqual(batch["labels"][1, -1].item(), -100)  # Padding value
        self.assertEqual(batch["input_ids"][1, -1].item(), 0)  # Padding token ID
        self.assertEqual(batch["attention_mask"][1, -1].item(), 0)  # Attention mask for padding


if __name__ == "__main__":
    unittest.main()