#FrootiNet 
#@author: megamind
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models 

class FrootiNet(nn.Module):

    def __init__(self):
        
        self.network = nn.Sequential(
            nn.Conv2d(3,64,3, padding=1),
            nn.Conv2d(64,64,3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(3,138,3, padding=1),
            nn.Conv2d(64,128,3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(64,128,3, padding=1),
            nn.Conv2d(128,128,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128,256,1),
            nn.Conv2d(256,512,25),
            nn.Conv2d(512,128,1),
            nn.Conv2d(128,60,1),
            nn.Softmax()
        )
    
    def forward(self,x):
        x = self.network(x)
