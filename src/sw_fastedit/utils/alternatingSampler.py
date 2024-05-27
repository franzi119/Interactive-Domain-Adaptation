import torch
from torch.utils.data import Dataset, Sampler

class AlternatingSampler(Sampler):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.num_samples = len(dataset1) + len(dataset2)
        #print("num_samples:", self.num_samples)
        self.indices1 = torch.randperm(len(self.dataset1)).tolist()
        #print('indices1',self.indices1)
        self.indices2 = torch.randperm(len(self.dataset2)).tolist()
        #print('indices2',self.indices2)

    def __iter__(self):
        # Initialize counters for each dataset


        for i in range(self.num_samples):
            if i % 2 == 0:
                yield self.indices2[i % len(self.dataset2)]
            else:
                yield self.indices1[i % len(self.dataset1)]
    def __len__(self):
        return self.num_samples

class BatchSampler(Sampler):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2