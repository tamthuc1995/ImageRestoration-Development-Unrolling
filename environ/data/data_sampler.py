import math
import torch
from torch.utils.data.sampler import Sampler


class ResumeableSampler(Sampler):
    """
    A sampling deterministic sampling process and resume 
        at certain iteration in a epoch
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0
        self.current_sample = -1
        self.num_samples = len(self.dataset)

    def __iter__(self): 

        for sample_i in range(self.num_samples):
            if sample_i > self.current_sample:
                self.current_sample += 1
                yield sample_i

    def __len__(self):
        return self.num_samples

    def set_epoch_and_current_sample(self, current_epoch, current_sample):
        self.current_epoch = current_epoch
        self.current_sample = current_sample
        self.dataset.random_permute(seed=2024 + current_epoch)

