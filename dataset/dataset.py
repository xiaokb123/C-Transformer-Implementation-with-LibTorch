# 数据集定义
# datasets/dataset.py
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return len(self.enc_inputs)

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]