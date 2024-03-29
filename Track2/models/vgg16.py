import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class VGG16(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(VGG16, self).__init__()
        
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.block_3 = nn.Sequential(        
            nn.Conv2d(128, 256, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1),
            nn.ReLU(),        
            nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
          
        self.block_4 = nn.Sequential(   
            nn.Conv2d(256, 512, (3, 3), (1, 1), padding=1),
            nn.ReLU(),        
            nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1),
            nn.ReLU(),        
            nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.block_5 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), (1, 1),padding=1),
            nn.ReLU(),            
            nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1),
            nn.ReLU(),            
            nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1),
            nn.ReLU(),    
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))             
        )
            
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
            
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()
                    
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)

        return logits, probas