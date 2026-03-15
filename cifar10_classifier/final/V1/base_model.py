import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.init as init
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchinfo import summary
import seaborn as sns
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
import os

class BaseModelF32(nn.Module):
    def __init__(self, blocks, final_head):
        super(BaseModelF32, self).__init__()
        self.blocks = nn.ModuleList(blocks)
        self.final_head = final_head
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_head(x)
        return x