import unittest
from pathlib import Path
from scripts.predict_llama import main

class TestPredictLlama(unittest.TestCase):
    def setUp(self):
        self.test_model = "yujiepan/llama-3.3-tiny-random"
        self.base_path = Path(__file__).parent.parent
        self.predictions_path = self.base_path / 'data' / 'evaluation' / 'llm_predictions' / f"{self.test_model.replace('/', '_')}.jsonl"
        
        # Clean up any existing prediction file
        if self.predictions_path.exists():
            self.predictions_path.unlink()

    def test_predict_llama(self):
        # Run prediction
        main(self.test_model)
        
        # Check that predictions file was created
        self.assertTrue(self.predictions_path.exists())
        
        # Basic validation of predictions file
        with open(self.predictions_path) as f:
            first_line = f.readline()
            self.assertTrue(len(first_line) > 0)
            self.assertTrue('"id":' in first_line)
            self.assertTrue('"prompt":' in first_line)
            self.assertTrue('"prediction":' in first_line)

    def tearDown(self):
        # Clean up prediction file after test
        if self.predictions_path.exists():
            self.predictions_path.unlink()

if __name__ == '__main__':
    unittest.main()
