from pathlib import Path
from unittest import TestCase
import jsonlines


class GoldLabelsTestCase(TestCase):

    def setUp(self):
        self.gold_file = Path(__file__).parent.parent / 'data' / 'evaluation' / 'gold_labels' / 'gold.jsonl'
        self.assertTrue(self.gold_file.exists())
        self.predictions_dir = Path(__file__).parent.parent / 'data' / 'evaluation' / 'llm_predictions'
        self.assertTrue(self.predictions_dir.exists())

    def test_gold_data_structure(self):
        with jsonlines.open(self.gold_file) as reader:
            data = list(reader)
            self.assertEqual(len(data), 1000, "Gold file does not have 1000 lines")

            for item in data:
                required_fields = ['text_a', 'text_b', 'labels_a', 'labels_b', 'subset', 'id']
                for field in required_fields:
                    self.assertIn(field, item, f"Field '{field}' missing in gold data")
                
                # Check that labels are lists of numbers
                self.assertTrue(isinstance(item['labels_a'], list))
                self.assertTrue(isinstance(item['labels_b'], list))
                self.assertTrue(all(isinstance(x, (int, float)) for x in item['labels_a']))
                self.assertTrue(all(isinstance(x, (int, float)) for x in item['labels_b']))

    def test_matching_ids(self):
        # Load gold data ids
        with jsonlines.open(self.gold_file) as reader:
            gold_ids = [item['id'] for item in reader]

        # Check one predictions file
        pred_file = next(self.predictions_dir.glob('*.jsonl'))
        with jsonlines.open(pred_file) as reader:
            pred_ids = [item['item_id'] for item in reader]
            self.assertEqual(
                gold_ids,
                pred_ids,
                f"IDs in {pred_file.name} do not match gold data IDs"
            )

    def test_load_gold_data(self):
        from evaluation.utils import load_gold_data
        
        # Load the gold data
        gold_samples = load_gold_data(self.gold_file)
        
        # Basic checks
        self.assertEqual(len(gold_samples), 1000)
        
        # Check sample structure
        sample = gold_samples[0]
        self.assertTrue(hasattr(sample, 'tokens_a'))
        self.assertTrue(hasattr(sample, 'tokens_b'))
        self.assertTrue(hasattr(sample, 'labels_a'))
        self.assertTrue(hasattr(sample, 'labels_b'))
        
        # Check types
        self.assertTrue(isinstance(sample.tokens_a, tuple))
        self.assertTrue(isinstance(sample.tokens_b, tuple))
        self.assertTrue(isinstance(sample.labels_a, tuple))
        self.assertTrue(isinstance(sample.labels_b, tuple))

        # for sample in gold_samples:
        #     self.assertEqual(len(sample.tokens_a), len(sample.labels_a))
        #     self.assertEqual(len(sample.tokens_b), len(sample.labels_b))
