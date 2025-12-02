import itertools
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List
import datasets
import os
import sys
import permpy
from scipy.stats import spearmanr

from rsd.data import DifferenceDataset
from rsd.recognizers.base import DifferenceRecognizer
from rsd.recognizers.utils import DifferenceSample

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
from scripts.project_labels import project_labels


@dataclass
class DifferenceRecognitionResult:
    spearman: float

    def __repr__(self):
        return f"{self.spearman:.3f}"


class DifferenceRecognitionBenchmark:

    def __init__(self,
                 positive_dataset: DifferenceDataset,
                 negative_dataset: DifferenceDataset = None,
                 positive_ratio: float = 1.0,
                 num_sentences_per_document: int = 1,
                 num_inversions: int = 0,
                 seed: int = 42,
                 pl: bool = False,
                 ):
        self.positive_ratio = positive_ratio
        assert 0 <= self.positive_ratio <= 1
        if positive_ratio < 1:
            assert negative_dataset is not None
        self.num_sentences_per_document = num_sentences_per_document
        self.num_inversions = num_inversions
        assert 0 <= self.num_inversions <= self.num_sentences_per_document * (self.num_sentences_per_document - 1) / 2
        self.seed = seed
        self.random = random.Random(self.seed)
        self.positive_dataset = positive_dataset
        self.negative_dataset = negative_dataset
        self.pl = pl
        self._load_sentences()
        self._build_documents()

    def _load_sentences(self):
        positive_sentences = self.positive_dataset.get_samples()
        self.sentences = []
        self.sentences += positive_sentences
        if self.negative_dataset is not None:
            negative_sentences = self.negative_dataset.get_samples()
            num_sentences = int(len(positive_sentences) / self.positive_ratio)
            num_negatives = num_sentences - len(positive_sentences)
            self.sentences += negative_sentences[:num_negatives]
            if num_negatives > len(negative_sentences):
                for _ in range(num_negatives - len(negative_sentences)):
                    self.sentences.append(deepcopy(self.random.choice(negative_sentences)))
        self.random.shuffle(self.sentences)

    def _build_documents(self):
        self.documents: List[DifferenceSample] = []
        num_documents = math.floor(len(self.sentences) / self.num_sentences_per_document)
        if self.num_inversions > 0:
            self.permutations = [p for p in permpy.PermSet.all(self.num_sentences_per_document) if
                                 p.num_inversions() == self.num_inversions]
        else:
            self.permutations = None
        for i in range(num_documents):
            sentences = self.sentences[i * self.num_sentences_per_document:(i + 1) * self.num_sentences_per_document]
            if self.pl:
                for sentence in sentences:
                    print(type(sentence))
                sentences = [project_labels(sentence, self.positive_dataset.tgt_lang) for sentence in sentences]
            document = DifferenceSample(
                tokens_a=tuple(itertools.chain.from_iterable([sentence.tokens_a for sentence in sentences])),
                tokens_b=tuple(itertools.chain.from_iterable([sentence.tokens_b for sentence in sentences])),
                labels_a=tuple(itertools.chain.from_iterable([sentence.labels_a for sentence in sentences])),
                labels_b=tuple(itertools.chain.from_iterable([sentence.labels_b for sentence in sentences])),
            )
            if self.permutations is not None:
                permutation = self.random.choice(self.permutations)
                sentences = [sentences[i] for i in permutation]  # Apply permutation
                document.tokens_b = tuple(itertools.chain.from_iterable([sentence.tokens_b for sentence in sentences]))
                document.labels_b = tuple(itertools.chain.from_iterable([sentence.labels_b for sentence in sentences]))
            self.documents.append(document)

    def __str__(self):
        return f"DifferenceRecognitionBenchmark(pos={self.positive_dataset}, neg={self.negative_dataset}, " \
                f"positive_ratio={self.positive_ratio}, num_sentences_per_document={self.num_sentences_per_document}, " \
                f"num_inversions={self.num_inversions}, seed={self.seed}"

    def evaluate(self,
                 recognizer: DifferenceRecognizer,
                 predict_kwargs: Dict = None,
                 ) -> DifferenceRecognitionResult:
        if predict_kwargs is None:
            predict_kwargs = {}
        predictions = recognizer.predict_all(
            a=[" ".join(sample.tokens_a) for sample in self.documents],
            b=[" ".join(sample.tokens_b) for sample in self.documents],
            **predict_kwargs,
        )
        assert len(predictions) == len(self.documents)
        gold_labels = []
        predicted_labels = []
        for sample, prediction in zip(self.documents, predictions):
            sample_gold_labels = list(sample.labels_a)
            sample_predicted_labels = list(prediction.labels_a)
            if set(sample.labels_b) != {-1}:
                sample_gold_labels += list(sample.labels_b)
                sample_predicted_labels += list(prediction.labels_b)
            assert len(sample_gold_labels) == len(sample_predicted_labels), f"Differing number of labels in sample {sample}: {len(sample_gold_labels)} vs. {len(sample_predicted_labels)}"
            sample_predicted_labels = [label for label, gold_label in zip(sample_predicted_labels, sample_gold_labels) if gold_label != -1]
            sample_gold_labels = [label for label in sample_gold_labels if label != -1]
            gold_labels += sample_gold_labels
            predicted_labels += sample_predicted_labels
        assert len(gold_labels) == len(predicted_labels)
        spearman = spearmanr(
            a=gold_labels,
            b=predicted_labels,
        ).correlation
        return DifferenceRecognitionResult(
            spearman=spearman,
        )

    @property
    def num_document_pairs(self) -> int:
        return len(self.documents)

    @property
    def num_tokens(self) -> int:
        num_tokens = 0
        for document in self.documents:
            num_tokens += len(document.tokens_a)
            num_tokens += len(document.tokens_b)
        return num_tokens

    @property
    def num_labels_lt_05(self) -> int:
        num_labels = 0
        for document in self.documents:
            num_labels += len([label for label in document.labels_a if 0 <= label < 0.5])
            num_labels += len([label for label in document.labels_b if 0 <= label < 0.5])
        return num_labels

    @property
    def num_labels_gte_05(self) -> int:
        num_labels = 0
        for document in self.documents:
            num_labels += len([label for label in document.labels_a if label >= 0.5])
            num_labels += len([label for label in document.labels_b if label >= 0.5])
        return num_labels

    @property
    def num_unlabeled_tokens(self) -> int:
        num_unlabeled_tokens = 0
        for document in self.documents:
            num_unlabeled_tokens += len([label for label in document.labels_a if label == -1])
            num_unlabeled_tokens += len([label for label in document.labels_b if label == -1])
        return num_unlabeled_tokens

    def to_dataset(self, both_directions: bool = False) -> datasets.Dataset:
        """Convert the benchmark documents to a HuggingFace dataset.
        
        Args:
            both_directions (bool, optional): If True, each document is added twice with the order of the text and labels swapped.
                                               Defaults to False.
        
        Returns:
            datasets.Dataset: Dataset containing text pairs and their difference labels.
        """
        data = {
            "text_a": [],
            "text_b": [],
            "labels_a": [],
            "labels_b": [],
        }
        
        for doc in self.documents:
            data["text_a"].append(" ".join(doc.tokens_a))
            data["text_b"].append(" ".join(doc.tokens_b))
            data["labels_a"].append(list(doc.labels_a))
            data["labels_b"].append(list(doc.labels_b))
            
            # If both_directions is True, add the document with swapped fields.
            if both_directions:
                data["text_a"].append(" ".join(doc.tokens_b))
                data["text_b"].append(" ".join(doc.tokens_a))
                data["labels_a"].append(list(doc.labels_b))
                data["labels_b"].append(list(doc.labels_a))
                
        return datasets.Dataset.from_dict(data)


class MultiLengthDifferenceRecognitionBenchmark:

    def __init__(self,
                 positive_dataset: DifferenceDataset,
                 negative_dataset: DifferenceDataset = None,
                 positive_ratio: float = 1.0,
                 max_sentences_per_document: int = 1,
                 max_inversions: int = 0,
                 seed: int = 42,
                 pl: bool = False,
                 ):
        assert max_sentences_per_document >= 1
        assert max_inversions <= max_sentences_per_document
        self.num_sentences_range = list(range(1, max_sentences_per_document + 1))
        if max_inversions == 0:
            self.num_inversions_range = [0] * len(self.num_sentences_range)
        elif max_inversions == max_sentences_per_document:
            self.num_inversions_range = [0, 1] + list(range(3, max_inversions + 1))
        else:
            raise NotImplementedError
        print(f"Combining {self.num_sentences_range} sentence counts with {self.num_inversions_range} inversion counts.")
        assert len(self.num_inversions_range) == len(self.num_sentences_range)
        self.benchmarks = []
        for num_sentences, num_inversions in zip(self.num_sentences_range, self.num_inversions_range):
            benchmark = DifferenceRecognitionBenchmark(
                positive_dataset=positive_dataset,
                negative_dataset=negative_dataset,
                positive_ratio=positive_ratio,
                num_sentences_per_document=num_sentences,
                num_inversions=num_inversions,
                seed=seed,
                pl=pl,
            )
            self.benchmarks.append(benchmark)

    def to_dataset(self, both_directions: bool = False) -> datasets.Dataset:
        ds = [benchmark.to_dataset(both_directions=both_directions) for benchmark in self.benchmarks]
        return datasets.concatenate_datasets(ds)
