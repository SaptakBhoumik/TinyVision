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

def post_conv0(channels):
    return nn.Sequential(
        nn.BatchNorm2d(channels),
        nn.PReLU(),
    )

class Model0(nn.Module):
    def __init__(self, norm, pool, post_conv,ratio = 2):
        super(Model0, self).__init__()
        self.norm = norm
        self.post_conv = post_conv
        self.pool = pool
        self.ratio = ratio

        self.conv0 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size = 7, padding = 3, groups = 10), # Each feature is processed separately and 2 more channels are created.for each
            self.post_conv(20),
            self.pool(20),

            #We now combine the 2 channels into 1 channel for each feature
            nn.Conv2d(20, 10, kernel_size = 7, padding = 3, groups = 10), 
            self.post_conv(10),
        )

        self.conv_block0_0 = self.create_conv_block0(2)#Handle feature generated from BW image and OSTU image and combine into 1 channel
        self.conv_block0_1 = self.create_conv_block0(2)#Do the same with 2 canny images
        self.conv_block0_2 = self.create_conv_block0(2)#Do the same with 2 scharr images
        self.conv_block0_3 = self.create_conv_block0(2)#Do the same with 2 lbd images
        self.conv_block0_4 = self.create_conv_block0(2)#Do the same with 2 gabor images

        self.conv1 = nn.Sequential(
            # Now we finally combine every result from every feature
            nn.Conv2d(5, int(5*self.ratio), kernel_size=3, padding=1),
            self.post_conv(int(5*self.ratio)),
            self.pool(int(5*self.ratio)),

            nn.Conv2d(int(5*self.ratio), 5, kernel_size=3, padding=1),
            self.post_conv(5),


            nn.Conv2d(5, int(5*self.ratio), kernel_size=3, padding=1), 
            self.post_conv(int(5*self.ratio)),
            self.pool(int(5*self.ratio)),

            nn.Conv2d(int(5*self.ratio), 5, kernel_size=3, padding=1),
            self.post_conv(5),


            nn.Conv2d(5, int(5*self.ratio), kernel_size=3, padding=1),
            self.post_conv(int(5*self.ratio)),
            self.pool(int(5*self.ratio)),

            nn.Conv2d(int(5*self.ratio), 5, kernel_size=3, padding=1),
            self.post_conv(5),
        )

        self.head = nn.Sequential(
            #Linear layer
            nn.Flatten(),

            nn.Linear(20, 10),  
            nn.PReLU(),

            nn.Linear(10, 5),  
            nn.PReLU(),
            
            nn.Linear(5, 2)  # Final output layer for binary classification
        )

    def create_conv_block0(self,channels):
        return nn.Sequential(
            nn.Conv2d(channels, int(channels*self.ratio*2), kernel_size=3, padding=1),
            self.post_conv(int(channels*self.ratio*2)),
            self.pool(int(channels*self.ratio*2)),

            nn.Conv2d(int(channels*self.ratio*2),int(channels*self.ratio), kernel_size=3, padding=1), 
            self.post_conv(int(channels*self.ratio)),
            
            
            nn.Conv2d(int(channels*self.ratio), int(self.ratio), kernel_size=3, padding=1), 
            self.post_conv(int(self.ratio)),
            self.pool(int(self.ratio)),

            nn.Conv2d(int(self.ratio), 1, kernel_size=3, padding=1),
            self.post_conv(1),
        )
    def forward(self, x):
        x = self.norm(x)
        x = self.conv0(x)

        x0 = torch.cat(
            [
                self.conv_block0_0(x[:,0:2,:,:]),
                self.conv_block0_1(x[:,2:4,:,:]),
                self.conv_block0_2(x[:,4:6,:,:]),
                self.conv_block0_3(x[:,6:8,:,:]),
                self.conv_block0_4(x[:,8:10,:,:]),
            ], dim=1)

        x1 = self.conv1(x0)
        y = self.head(x1)

        return y
    

class Model1(nn.Module):
    def __init__(self, norm, post_conv,ratio = 2):
        super(Model1, self).__init__()
        self.norm = norm
        self.post_conv = post_conv
        self.ratio = ratio

        self.conv0 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size = 7, padding = 3, groups = 10,stride=2),  # Each feature is processed separately and 2 more channels are created.for each
            self.post_conv(20),

            #We now combine the 2 channels into 1 channel for each feature
            nn.Conv2d(20, 10, kernel_size = 7, padding = 3, groups = 10), 
            self.post_conv(10),
        )

        self.conv_block0_0 = self.create_conv_block0(2)#Handle feature generated from BW image and OSTU image and combine into 1 channel
        self.conv_block0_1 = self.create_conv_block0(2)#Do the same with 2 canny images
        self.conv_block0_2 = self.create_conv_block0(2)#Do the same with 2 scharr images
        self.conv_block0_3 = self.create_conv_block0(2)#Do the same with 2 lbd images
        self.conv_block0_4 = self.create_conv_block0(2)#Do the same with 2 gabor images

        self.conv1 = nn.Sequential(
            # Now we finally combine every result from every feature
            nn.Conv2d(5, int(5*self.ratio), kernel_size=3, padding=1,stride=2),
            self.post_conv(int(5*self.ratio)),

            nn.Conv2d(int(5*self.ratio), 5, kernel_size=3, padding=1),
            self.post_conv(5),


            nn.Conv2d(5, int(5*self.ratio), kernel_size=3, padding=1,stride=2), 
            self.post_conv(int(5*self.ratio)),

            nn.Conv2d(int(5*self.ratio), 5, kernel_size=3, padding=1),
            self.post_conv(5),


            nn.Conv2d(5, int(5*self.ratio), kernel_size=3, padding=1,stride=2),
            self.post_conv(int(5*self.ratio)),

            nn.Conv2d(int(5*self.ratio), 5, kernel_size=3, padding=1),
            self.post_conv(5),
        )

        self.head = nn.Sequential(
            #Linear layer
            nn.Flatten(),

            nn.Linear(20, 10),  
            nn.PReLU(),

            nn.Linear(10, 5),  
            nn.PReLU(),
            
            nn.Linear(5, 2)  # Final output layer for binary classification
        )

    def create_conv_block0(self,channels):
        return nn.Sequential(
            nn.Conv2d(channels, int(channels*self.ratio*2), kernel_size=3, padding=1,stride=2), 
            self.post_conv(int(channels*self.ratio*2)),

            nn.Conv2d(int(channels*self.ratio*2),int(channels*self.ratio), kernel_size=3, padding=1), 
            self.post_conv(int(channels*self.ratio)),
            
            
            nn.Conv2d(int(channels*self.ratio), int(self.ratio), kernel_size=3, padding=1,stride=2), 
            self.post_conv(int(self.ratio)),

            nn.Conv2d(int(self.ratio), 1, kernel_size=3, padding=1),
            self.post_conv(1),
        )
    def forward(self, x):
        x = self.norm(x)
        x = self.conv0(x)

        x0 = torch.cat(
            [
                self.conv_block0_0(x[:,0:2,:,:]),
                self.conv_block0_1(x[:,2:4,:,:]),
                self.conv_block0_2(x[:,4:6,:,:]),
                self.conv_block0_3(x[:,6:8,:,:]),
                self.conv_block0_4(x[:,8:10,:,:]),
            ], dim=1)

        x1 = self.conv1(x0)
        y = self.head(x1)

        return y
    

class Model2(nn.Module):
    def __init__(self, norm, post_conv):
        super(Model2, self).__init__()
        self.norm = norm
        self.post_conv = post_conv

        self.conv0 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size = 7, padding = 3, groups = 10, stride = 2),  # Each feature is processed separately and 2 more channels are created.for each
            self.post_conv(20),

            #We now combine the 2 channels into 1 channel for each feature
            nn.Conv2d(20, 10, kernel_size = 7, padding = 3, groups = 10), 
            self.post_conv(10),
        )

        self.conv_block0_0 = self.create_conv_block0(2)#Handle feature generated from BW image and OSTU image and combine into 1 channel
        self.conv_block0_1 = self.create_conv_block0(2)#Do the same with 2 canny images
        self.conv_block0_2 = self.create_conv_block0(2)#Do the same with 2 scharr images
        self.conv_block0_3 = self.create_conv_block0(2)#Do the same with 2 lbd images
        self.conv_block0_4 = self.create_conv_block0(2)#Do the same with 2 gabor images

        self.conv1 = nn.Sequential(
            # Now we finally combine every result from every feature
            nn.Conv2d(5, 5, kernel_size=3, padding=1, stride=2), #(16x16) -> (8x8)
            self.post_conv(5),

            nn.Conv2d(5, 5, kernel_size=2, stride=2), #(8x8) -> (4x4)
            self.post_conv(5),

            nn.Conv2d(5, 5, kernel_size=2, stride=2), #(4x4) -> (2x2)
            self.post_conv(5),
        )

        self.head = nn.Sequential(
            #Linear layer
            nn.Flatten(),

            nn.Linear(20, 10),  
            nn.PReLU(),

            nn.Linear(10, 5),  
            nn.PReLU(),
            
            nn.Linear(5, 2)  # Final output layer for binary classification
        )

    def create_conv_block0(self,channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2), #(64x64) -> (32x32)
            self.post_conv(channels),
            
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=2), #(32x32) -> (16x16)
            self.post_conv(1),
        )
    def forward(self, x):
        x = self.norm(x)
        x = self.conv0(x)

        x0 = torch.cat(
            [
                self.conv_block0_0(x[:,0:2,:,:]),
                self.conv_block0_1(x[:,2:4,:,:]),
                self.conv_block0_2(x[:,4:6,:,:]),
                self.conv_block0_3(x[:,6:8,:,:]),
                self.conv_block0_4(x[:,8:10,:,:]),
            ], dim=1)

        x1 = self.conv1(x0)
        y = self.head(x1)

        return y


class Model3(nn.Module):
    def __init__(self, norm, post_conv,pool):
        super(Model3, self).__init__()
        self.norm = norm
        self.post_conv = post_conv
        self.pool = pool

        self.conv0 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size = 7, padding = 3, groups = 10), # Each feature is processed separately and 2 more channels are created.for each
            self.post_conv(20),
            self.pool(20),

            # We now combine the 2 channels into 1 channel for each feature
            nn.Conv2d(20, 10, kernel_size = 7, padding = 3, groups = 10), 
            self.post_conv(10),
        )

        self.conv_block0_0 = self.create_conv_block0(2)#Handle feature generated from BW image and OSTU image and combine into 1 channel
        self.conv_block0_1 = self.create_conv_block0(2)#Do the same with 2 canny images
        self.conv_block0_2 = self.create_conv_block0(2)#Do the same with 2 scharr images
        self.conv_block0_3 = self.create_conv_block0(2)#Do the same with 2 lbd images
        self.conv_block0_4 = self.create_conv_block0(2)#Do the same with 2 gabor images

        self.conv1 = nn.Sequential(
            # Now we finally combine every result from every feature
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            self.post_conv(5),
            self.pool(5),

            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            self.post_conv(5),
            self.pool(5),

            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            self.post_conv(5),
            self.pool(5),
        )

        self.head = nn.Sequential(
            # Linear layer
            nn.Flatten(),

            nn.Linear(20, 10),  
            nn.PReLU(),

            nn.Linear(10, 5),  
            nn.PReLU(),
            
            nn.Linear(5, 2)  # Final output layer for binary classification
        )

    def create_conv_block0(self,channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), #(64x64) -> (32x32)
            self.post_conv(channels),
            self.pool(channels),
            
            nn.Conv2d(channels, 1, kernel_size=3, padding=1), #(32x32) -> (16x16)
            self.post_conv(1),
            self.pool(1),
        )
    def forward(self, x):
        x = self.norm(x)
        x = self.conv0(x)

        x0 = torch.cat(
            [
                self.conv_block0_0(x[:,0:2,:,:]),
                self.conv_block0_1(x[:,2:4,:,:]),
                self.conv_block0_2(x[:,4:6,:,:]),
                self.conv_block0_3(x[:,6:8,:,:]),
                self.conv_block0_4(x[:,8:10,:,:]),
            ], dim=1)

        x1 = self.conv1(x0)
        y = self.head(x1)

        return y