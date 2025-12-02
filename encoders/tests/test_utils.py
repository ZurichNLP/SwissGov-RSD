import unittest
import torch
from transformers import AutoTokenizer

from encoders.utils import format_as_sequence_pair


class UtilsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def test_format_as_sequence_pair(self):
        example = {
            "text_a": "This is text A",
            "text_b": "This is sentence B",
            "labels_b": [0, 0, 0.5, 1]  # Label for each word in text_b
        }

        result = format_as_sequence_pair(example, self.tokenizer)

        print(result)

        # Check that we got the expected keys
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)

        # Check shapes match
        self.assertEqual(result["input_ids"].shape, result["attention_mask"].shape)
        self.assertEqual(result["input_ids"].shape, result["labels"].shape)

        input_ids = result["input_ids"].tolist()
        self.assertEqual(self.tokenizer.decode(input_ids), f"{self.tokenizer.cls_token} this is text a {self.tokenizer.sep_token} this is sentence b {self.tokenizer.sep_token}")
        seq_b_start = 5

        labels = result["labels"]

        # Check text_a portion has -100 labels
        self.assertTrue(torch.all(labels[:seq_b_start] == -100))

        # Check special tokens have -100 labels
        special_token_positions = [i for i, token_id in enumerate(input_ids)
                                 if token_id in [self.tokenizer.bos_token_id,
                                               self.tokenizer.eos_token_id,
                                               self.tokenizer.pad_token_id]]
        for pos in special_token_positions:
            self.assertEqual(labels[pos].item(), -100)

        # Check that the expected label sequence appears in the labels
        labels_list = labels.tolist()
        found_sequence = False
        for i in range(len(labels_list) - 3):
            if labels_list[i:i+4] == [0, 0, 0.5, 1]:
                found_sequence = True
                break
        self.assertTrue(found_sequence, f"Expected label sequence not found in labels {labels_list}")


if __name__ == "__main__":
    unittest.main() 