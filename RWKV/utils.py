from torch.utils.data import Dataset
import numpy as np
from convert import file_header
import os
import torch

class BinIdxDataset(Dataset):
    def __init__(self, idx_path, ctx_len, miniepoch_size):
        self.ctx_len = ctx_len
        bin_path = os.path.splitext(idx_path)[0] + '.bin'
        self.bin_memoryview = memoryview(np.memmap(bin_path, mode = 'r'))
        self.idx_memoryview = memoryview(np.memmap(idx_path, mode = 'r'))
        self.header = file_header()
        with open(idx_path, 'rb') as f:
            self.header.unpack(f.read(64))
        self.sizes = np.frombuffer(self.idx_memoryview, dtype = np.uint64, count = self.header.num_sizes, offset = 64).astype(np.int64)
        self.pointer = np.cumsum(self.sizes)
        self.total_size = self.header.total_len
        self.single_size = np.array([1], dtype = self.header.dtype).itemsize
        self.miniepoch_size = miniepoch_size

    def __getitem__(self, idx):
        i = np.random.randint(0, self.total_size - (self.ctx_len + 1))
        data = np.frombuffer(self.bin_memoryview, dtype = self.header.dtype, offset = self.single_size * i, count = self.ctx_len + 1).astype(int)
        return torch.tensor(data[:-1]), torch.tensor(data[1:])
    
    def __len__(self):
        return self.miniepoch_size
    
    def get_one_passage(self):
        i = np.random.randint(0, self.sizes.shape[0])
        return torch.tensor(np.frombuffer(self.bin_memoryview, dtype = self.header.dtype, offset = self.single_size * (self.pointer[i] - self.sizes[i]),
                                          count = self.sizes[i]).astype(int))