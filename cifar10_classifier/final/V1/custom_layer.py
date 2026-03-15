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

class HybridPoolingV0(nn.Module):
    def __init__(self,channels,pool1 = nn.MaxPool2d, pool2 = nn.AvgPool2d):
        super(HybridPoolingV0, self).__init__()
        self.pool1 = pool1(2)
        self.pool2 = None 
        if pool2 is not None:
            self.pool2 = pool2(2)
        self.activation = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        pooled1 = self.pool1(x)
        if self.pool2 == None:
            return self.activation(pooled1)*pooled1
            
        pooled2 = self.pool2(x)
        return self.activation(pooled1)*pooled2