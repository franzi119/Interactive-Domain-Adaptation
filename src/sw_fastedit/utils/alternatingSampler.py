import torch
from torch.utils.data import Dataset, Sampler

class AlternatingSampler(Sampler):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.epoch_length = len(dataset1)*2

    def __iter__(self):
        indices1 = torch.randperm(len(self.dataset1)).tolist()
        #add length of first dataset to the second one to avoid overlapping indices
        indices2 = (torch.randperm(len(self.dataset2))+len(self.dataset1)).tolist()
        #print(indices1)
        temp2 = indices2.copy()
        #print(temp2)

        for i in range(self.epoch_length):
            if i % 2 == 0:
                yield indices1.pop(0)
            else:
                if(len(temp2) == 0):
                    temp2 = indices2
                yield temp2.pop(0)

    def __len__(self):
        return self.epoch_length