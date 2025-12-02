import unittest

from transformers import AutoTokenizer

from encoders.finetuning.data import load_encoder_training_data


class TestDataCreation(unittest.TestCase):

    def test_create_training_dataset(self):
        # Test basic dataset creation
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dataset = load_encoder_training_data(tokenizer)
        
        # Check that we have both splits
        self.assertIn('train', dataset)
        self.assertIn('eval', dataset)
        
        # Check that the splits are roughly equal size
        train_size = len(dataset['train'])
        eval_size = len(dataset['eval'])
        self.assertGreater(train_size, 0)
        self.assertGreater(eval_size, 0)
        # Allow for rounding differences
        self.assertLessEqual(abs(train_size - eval_size), 1)
        
        # Check that the dataset has the expected columns
        expected_columns = ['text_a', 'text_b', 'labels_b']
        for column in expected_columns:
            self.assertIn(column, dataset['train'].column_names)
            self.assertIn(column, dataset['eval'].column_names)

        # Check that there is no overlap between train and eval
        train_ids = dataset['train'].unique('id')
        eval_ids = dataset['eval'].unique('id')
        self.assertFalse(set(train_ids) & set(eval_ids))

if __name__ == '__main__':
    unittest.main()
