import torch
from torch.utils.data import Sampler

class AlternatingSampler(Sampler):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1 #source dataset
        self.dataset2 = dataset2
        if len(dataset1) > len(dataset2):
            self.epoch_length = len(dataset1)*2
        else:
            self.epoch_length = len(dataset2)*2

    def __iter__(self):
        indices1 = torch.randperm(len(self.dataset1))
        #add length of first dataset to the second one to avoid overlapping indices
        indices2 = (torch.randperm(len(self.dataset2))+len(self.dataset1))

        j,k=0,0
        for i in range(self.epoch_length):
            if i % 2 == 0:
                yield indices1[j%len(indices1)].item()
                j = j + 1
            else:
                yield indices2[k%len(indices2)].item()
                #print('k moduo len',k%len(indices2))
                k = k + 1

    def __len__(self):
        return self.epoch_length
    

class SpecificSampler(Sampler):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        if len(dataset1) > len(dataset2):
            self.epoch_length = len(dataset1)*2
        else:
            self.epoch_length = len(dataset2)*2

    def __iter__(self):
        indices1 = torch.randperm(len(self.dataset1))

        for i in range(self.epoch_length):
                yield indices1[i%len(indices1)].item()

    def __len__(self):
        return self.epoch_length