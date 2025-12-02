from typing import List

from rsd.recognizers.utils import DifferenceSample

class DifferenceDataset:

    def __str__(self):
        raise NotImplemented

    def get_samples(self) -> List[DifferenceSample]:
        raise NotImplemented
