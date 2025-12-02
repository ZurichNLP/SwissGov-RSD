from pathlib import Path
from unittest import TestCase
import jsonlines

from evaluation.utils import load_predictions


class LLMPredictionDataTestCase(TestCase):

    def setUp(self):
        self.predictions_dir = Path(__file__).parent.parent / 'data' / 'evaluation' / 'llm_predictions'
        self.assertTrue(self.predictions_dir.exists())

    def test_data_complete(self):
        jsonl_files = list(self.predictions_dir.glob('*.jsonl'))
        self.assertGreater(len(jsonl_files), 0, "No JSONL files found")
        
        for file in jsonl_files:
            with jsonlines.open(file) as reader:
                data = list(reader)
                self.assertEqual(len(data), 1000, f"File {file.name} does not have 1000 lines")

    def test_consistent_item_ids(self):
        """Test that all prediction files have the same sequence of item IDs."""
        jsonl_files = list(self.predictions_dir.glob('*.jsonl'))
        self.assertGreater(len(jsonl_files), 0, "No JSONL files found")

        # Get item IDs from first file as reference
        with jsonlines.open(jsonl_files[0]) as reader:
            reference_ids = [item.get('item_id', item.get('id')) for item in reader]

        # Compare with all other files
        for file in jsonl_files[1:]:
            with jsonlines.open(file) as reader:
                current_ids = [item.get('item_id', item.get('id')) for item in reader]
                self.assertEqual(
                    reference_ids,
                    current_ids,
                    f"Item IDs in {file.name} do not match the sequence in {jsonl_files[0].name}"
                )

    def test_load_predictions(self):
        from evaluation.utils import load_predictions
        
        # Get first predictions file
        pred_file = next(self.predictions_dir.glob('*.jsonl'))
        
        # Load predictions
        predictions = load_predictions(pred_file)
        
        # Basic checks
        self.assertEqual(len(predictions), 1000)
        
        # Check prediction structure
        prediction = predictions[0]
        self.assertTrue(hasattr(prediction, 'item_id'))
        self.assertTrue(hasattr(prediction, 'sentence1'))
        self.assertTrue(hasattr(prediction, 'sentence2'))
        self.assertTrue(hasattr(prediction, 'api_request'))
        self.assertTrue(hasattr(prediction, 'api_response'))
        self.assertTrue(hasattr(prediction, 'provider'))
        
        # Check types
        self.assertTrue(isinstance(prediction.item_id, str))
        self.assertTrue(isinstance(prediction.sentence1, str))
        self.assertTrue(isinstance(prediction.sentence2, str))
        self.assertTrue(isinstance(prediction.api_request, dict) or isinstance(prediction.api_request, str))
        self.assertTrue(isinstance(prediction.api_response, dict) or isinstance(prediction.api_response, str))
        self.assertTrue(isinstance(prediction.provider, str))


class ParseLLMPredictionsTestCase(TestCase):

    def setUp(self):
        self.predictions_dir = Path(__file__).parent.parent / 'data' / 'evaluation' / 'llm_predictions'

    def test_json_str_extraction(self):
        jsonl_files = list(self.predictions_dir.glob('*.jsonl'))
        self.assertGreater(len(jsonl_files), 0, "No JSONL files found")
        
        invalid_counts = {}
        for pred_file in jsonl_files:
            num_invalid = 0
            predictions = load_predictions(pred_file)
            
            for i, prediction in enumerate(predictions):
                content = prediction.get_json_str()
                # Basic validation of extracted content
                self.assertIsInstance(content, str)
                self.assertTrue(content.strip())  # Not empty
                if not content.startswith("{"):
                    print(f"Invalid content in {pred_file.name}, line {i+1}: {content}")
                    num_invalid += 1
            
            invalid_counts[str(pred_file)] = num_invalid

        print()
        for file_path, count in invalid_counts.items():
            print(f"{Path(file_path).name}: {count} invalid content strings")

    def test_parse_json(self):
        jsonl_files = list(self.predictions_dir.glob('*.jsonl'))
        self.assertGreater(len(jsonl_files), 0, "No JSONL files found")
        invalid_counts = {}

        for pred_file in jsonl_files:
            predictions = load_predictions(pred_file)
            num_invalid = 0

            for i, prediction in enumerate(predictions):
                data = prediction.get_json()
                if not data:
                    print(f"Invalid JSON in {pred_file.name}, line {i+1}: {prediction.get_json_str()}")
                    num_invalid += 1
            invalid_counts[str(pred_file)] = num_invalid

        print()
        for file_path, count in invalid_counts.items():
            print(f"{Path(file_path).name}: {count} invalid JSON predictions")

    def test_get_difference_sample(self):
        jsonl_files = list(self.predictions_dir.glob('*.jsonl'))
        self.assertGreater(len(jsonl_files), 0, "No JSONL files found")

        for pred_file in jsonl_files:
            predictions = load_predictions(pred_file)

            for i, prediction in enumerate(predictions):
                sample = prediction.get_difference_sample()
                assert len(sample.tokens_a) == len(sample.labels_a)
                assert len(sample.tokens_b) == len(sample.labels_b)
