import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from torch.utils.data.dataset import ConcatDataset, Dataset


class ReDataset(Dataset):
    def __init__(self, data,data2,config=None,id2num = None):
        self.data = data
        self.config = config
        self.id2num = id2num
        self.mode = 0
        if data2:
            for i in range(len(data)):
                self.data[i]['tokens2'] = data2[i]['tokens']
            self.mode = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        idxs = [item.get('idx', 0) for item in data] 
        label = torch.tensor([self.id2num[item['relation']] if item['relation'] in self.id2num else 0 for item in data])
        tokens = [torch.tensor(item['tokens']) for item in data]
        tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
        strings = [item.get('string', 'None') for item in data] 
        is_pre_data = [item.get('is_pre_data', False) for item in data] 
        cluster = [item.get('cluster') for item in data]
        if not self.mode:
            return (idxs, label,tokens,strings, is_pre_data, cluster)
        else: 
            tokens2 = [torch.tensor(item['tokens2']) for item in data]
            tokens2 = nn.utils.rnn.pad_sequence(tokens2, batch_first=True, padding_value=0) 
            return (idxs, label,tokens,tokens2,strings, is_pre_data,cluster)
        
def get_data_loader(config,id2num, data, data2=None,shuffle=False, drop_last=False, batch_size=None, sampler=None):
    dataset = ReDataset(data,data2,config,id2num)

    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)
    return data_loader

