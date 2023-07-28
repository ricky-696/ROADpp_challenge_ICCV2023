import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Block(nn.Module):
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()

        group_width = cardinality * bottleneck_width        
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(self.expansion * group_width)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10, input_dim=3, input_size=[32, 32]):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        if len(num_blocks) > 3:
            self.layer4 = self._make_layer(num_blocks[3], 2)

        self.dropout = nn.Dropout(p = 0.5)

        self.classifier = nn.Sequential(
            nn.Linear(cardinality * bottleneck_width * 8 * 100, cardinality * bottleneck_width * 8),
            nn.ReLU(),
            nn.Linear(cardinality * bottleneck_width * 8, cardinality * bottleneck_width * 8),
            nn.ReLU(),
            nn.Linear(cardinality * bottleneck_width * 8, num_classes)
        )


    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out


def ResNeXt29_2x64d(num_class, input_dim, input_size):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=2, bottleneck_width=64, num_classes=num_class, input_dim=input_dim, input_size=input_size)

def ResNeXt29_4x64d(num_class, input_dim, input_size):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=4, bottleneck_width=64, num_classes=num_class, input_dim=input_dim, input_size=input_size)
    
def ResNeXt29_8x64d(num_class, input_dim, input_size):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=8, bottleneck_width=64, num_classes=num_class, input_dim=input_dim, input_size=input_size)
    
def ResNeXt29_32x4d(num_class, input_dim, input_size):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=32, bottleneck_width=4, num_classes=num_class, input_dim=input_dim, input_size=input_size)

def ResNeXt50_32x4d(num_class, input_dim, input_size):
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4, num_classes=num_class, input_dim=input_dim, input_size=input_size)

def ResNeXt101_32x4d(num_class, input_dim, input_size):
    return ResNeXt(num_blocks=[3, 4, 23, 3], cardinality=32, bottleneck_width=4, num_classes=num_class, input_dim=input_dim, input_size=input_size)

def ResNeXt101_64x4d(num_class, input_dim, input_size):
    return ResNeXt(num_blocks=[3, 4, 23, 3], cardinality=64, bottleneck_width=4, num_classes=num_class, input_dim=input_dim, input_size=input_size)


if __name__ == '__main__':
    model = ResNeXt101_64x4d(num_class=17, input_dim=30, input_size=[244, 244])
    x = torch.randn(2, 30, 360, 640)
    y = model(x)
    print(y.size())
    summary(model, (30, 32, 32), device='cpu')