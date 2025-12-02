import unittest
import json
from pathlib import Path

class TestDataSplit(unittest.TestCase):
    def setUp(self):
        self.base_path = Path(__file__).parent.parent
        self.train_path = self.base_path / "data" / "finetuning" / "llm_inputs" / "train.jsonl"
        self.valid_path = self.base_path / "data" / "finetuning" / "llm_inputs" / "valid.jsonl"

    def test_no_overlap_between_train_and_valid(self):
        # Read training data
        train_lines = set()
        with open(self.train_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                # We'll compare the actual text content
                if isinstance(data, dict):
                    if 'text' in data:
                        train_lines.add(data['text'])
                    elif 'input' in data:
                        train_lines.add(data['input'])
                else:
                    train_lines.add(line.strip())

        # Read validation data and check for overlap
        overlap_found = []
        with open(self.valid_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                text = None
                if isinstance(data, dict):
                    if 'text' in data:
                        text = data['text']
                    elif 'input' in data:
                        text = data['input']
                else:
                    text = line.strip()
                
                if text in train_lines:
                    overlap_found.append(text)

        self.assertEqual(len(overlap_found), 0, 
                        f"Found {len(overlap_found)} overlapping lines between train and validation sets. "
                        f"First overlapping example: {overlap_found[0] if overlap_found else ''}")

if __name__ == '__main__':
    unittest.main() 