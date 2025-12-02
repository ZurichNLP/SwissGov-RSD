from unittest import TestCase

from encoders.encoder_recognizer import EncoderDifferenceRecognizer


class EncoderDifferenceRecognizerTestCase(TestCase):

    def setUp(self) -> None:
        self.recognizer = EncoderDifferenceRecognizer("out/ModernBERT-base-rsd") # change path if needed

    def test_str(self):
        self.assertIn("EncoderRecognizer", str(self.recognizer))
        self.assertIn("out/ModernBERT-base-rsd", str(self.recognizer)) # change path if needed

    def test_predict(self):
        result = self.recognizer.predict(
            a="Chinese shares close higher Friday .",
            b="Chinese shares close lower Wednesday .",
        )
        self.assertEqual(('Chinese', 'shares', 'close', 'higher', 'Friday', '.'), result.tokens_a)
        self.assertEqual(('Chinese', 'shares', 'close', 'lower', 'Wednesday', '.'), result.tokens_b)
        self.assertEqual(6, len(result.labels_a))
        self.assertEqual(6, len(result.labels_b))
        self.assertNotEqual(result.labels_a, result.labels_b)
        self.assertIsInstance(result.labels_a[0], float)
        self.assertIsInstance(result.labels_b[0], float)
        print(result)

    def test_predict_with_subwords(self):
        result = self.recognizer.predict(
            a="This is a testtest sentence .",
            b="This is another testtest phrase .",
        )
        self.assertEqual(('This', 'is', 'a', 'testtest', 'sentence', '.'), result.tokens_a)
        self.assertEqual(('This', 'is', 'another', 'testtest', 'phrase', '.'), result.tokens_b)
        self.assertEqual(6, len(result.labels_a))
        self.assertEqual(6, len(result.labels_b))
        self.assertIsInstance(result.labels_a[0], float)
        self.assertIsInstance(result.labels_b[0], float)
