from unittest import TestCase
from evaluation.utils import parse_token_labels


class ParseTokenLabelsTestCase(TestCase):
    
    def test_parse_token_labels_example1(self):
        tokens1 = ["Iran", "hopes", "nuclear", "talks", "will", "yield", "`", "roadmap", "'"]
        tokens2 = ["Iran", "Nuclear", "Talks", "in", "Geneva", "Spur", "High", "Hopes"]
        predictions = {"sentence1": [["Iran", 5], ["hopes", 4], ["nuclear", 5], ["talks", 5], ["will", 3], ["yield", 3], ["`", -1], ["roadmap", 2], ["'", -1]], 
                       "sentence2": [["Iran", 5], ["Nuclear", 5], ["Talks", 5], ["in", 0], ["Geneva", 0], ["Spur", 3], ["High", 2], ["Hopes", 2]]}
        
        expected1 = [5.0, 4.0, 5.0, 5.0, 3.0, 3.0, -1.0, 2.0, -1.0]
        expected2 = [5.0, 5.0, 5.0, 0.0, 0.0, 3.0, 2.0, 2.0]
        result1 = parse_token_labels(tokens1, predictions["sentence1"])
        result2 = parse_token_labels(tokens2, predictions["sentence2"])
        
        self.assertEqual(expected1, result1)
        self.assertEqual(expected2, result2)

    def test_parse_token_labels_example2(self):
        tokens1 = ["Books", "To", "Help", "Kids", "Talk", "About", "Boston", "Marathon", "News"]
        tokens2 = ["Report", "of", "2", "explosions", "at", "finish", "line", "of", "Boston", "Marathon"]
        predictions = {"sentence1": [["Books", 1], ["To", 0], ["Help", 0], ["Kids", 0], ["Talk", 0], ["About", 4], ["Boston", 4], ["Marathon", 4], ["News", 4]],
                       "sentence2": [["Report", 1], ["of", 0], ["2", 0], ["explosions", 0], ["at", 0], ["finish", 0], ["line", 0], ["of", 4], ["Boston", 4], ["Marathon", 4]]}
        
        expected1 = [1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0]
        expected2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0]
        result1 = parse_token_labels(tokens1, predictions["sentence1"])
        result2 = parse_token_labels(tokens2, predictions["sentence2"])
        
        self.assertEqual(expected1, result1)
        self.assertEqual(expected2, result2)

    def test_parse_token_labels_example3(self):
        tokens1 = ["Chinese", "shares", "close", "lower", "Wednesday"]
        tokens2 = ["Chinese", "shares", "close", "higher", "Friday"]
        predictions = {"sentence1": [["Chinese", 5], ["shares", 5], ["close", 5], ["lower", 0], ["Wednesday", 3]],
                       "sentence2": [["Chinese", 5], ["shares", 5], ["close", 5], ["higher", 0], ["Friday", 3]]}
        
        expected1 = [5.0, 5.0, 5.0, 0.0, 3.0]
        expected2 = [5.0, 5.0, 5.0, 0.0, 3.0]
        result1 = parse_token_labels(tokens1, predictions["sentence1"])
        result2 = parse_token_labels(tokens2, predictions["sentence2"])
        
        self.assertEqual(expected1, result1)
        self.assertEqual(expected2, result2)

    def test_missing_tokens(self):
        tokens1 = ["Chinese", "shares", "close", "lower", "Wednesday"]
        tokens2 = ["Chinese", "shares", "close", "higher", "Friday"]
        predictions = {"sentence1": [["Chinese", 5], ["close", 5], ["lower", 0], ["Wednesday", 3]],
                       "sentence2": [["Chinese", 5], ["shares", 5], ["close", 5], ["higher", 0]]}

        expected1 = [5.0, 0.0, 5.0, 0.0, 3.0]
        expected2 = [5.0, 5.0, 5.0, 0.0, 0.0]
        result1 = parse_token_labels(tokens1, predictions["sentence1"])
        result2 = parse_token_labels(tokens2, predictions["sentence2"])

        self.assertEqual(expected1, result1)
        self.assertEqual(expected2, result2)

    def test_wrong_tokens(self):
        tokens1 = ["Chinese", "shares", "close", "lower", "Wednesday"]
        tokens2 = ["Chinese", "shares", "close", "higher", "Friday"]
        predictions = {"sentence1": [["Chinese", 5], ["xxxxx", 5], ["close", 5], ["lower", 0], ["Wednesday", 3]],
                       "sentence2": [["Chinese", 5], ["shares", 5], ["close", 5], ["higher", 0], ["xxxxx", 3]]}

        expected1 = [5.0, 0.0, 5.0, 0.0, 3.0]
        expected2 = [5.0, 5.0, 5.0, 0.0, 0.0]
        result1 = parse_token_labels(tokens1, predictions["sentence1"])
        result2 = parse_token_labels(tokens2, predictions["sentence2"])

        self.assertEqual(expected1, result1)
        self.assertEqual(expected2, result2)
