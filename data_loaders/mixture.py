import torch
from torch.utils.data import ConcatDataset, random_split
import random
import numpy as np

class MixtureDataset:
    """
    Combines two datasets with a given ratio. The ratio is enforced by
    sampling according to weights; this class returns a single ConcatDataset
    but you can also create a weighted sampler.
    For simplicity, we just concatenate after resizing the larger dataset
    to match the desired ratio.
    """
    @staticmethod
    def create(dataset_a, dataset_b, ratio_a, shuffle=True, seed=42):
        """
        ratio_a: float between 0 and 1, fraction from dataset_a.
        Returns a ConcatDataset.
        """
        len_a = len(dataset_a)
        len_b = len(dataset_b)
        # If ratio_a is 0.7, we want total size = len_a + len_b,
        # but we may need to subsample the larger dataset.
        # We'll keep all of the smaller dataset and subsample the larger.
        if ratio_a >= 0.5:
            # dataset_a is larger or equal weight
            target_a = int((len_b * ratio_a) / (1 - ratio_a)) if len_b > 0 else len_a
            target_a = min(target_a, len_a)
            indices_a = np.random.RandomState(seed).choice(len_a, target_a, replace=False)
            subset_a = torch.utils.data.Subset(dataset_a, indices_a)
            return ConcatDataset([subset_a, dataset_b])
        else:
            target_b = int((len_a * (1 - ratio_a)) / ratio_a) if len_a > 0 else len_b
            target_b = min(target_b, len_b)
            indices_b = np.random.RandomState(seed).choice(len_b, target_b, replace=False)
            subset_b = torch.utils.data.Subset(dataset_b, indices_b)
            return ConcatDataset([dataset_a, subset_b])