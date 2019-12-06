import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 20

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(16*128, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 16*128)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, nclasses)

    def forward(self, x):
        return self.model(x)

class ResNext50(nn.Module):
    def __init__(self):
        super(ResNext50, self).__init__()
        self.model = models.resnext50_32x4d(pretrained = True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, nclasses)
    def forward(self, x):
        return self.model(x)

class ResNext101(nn.Module):
    def __init__(self):
        super(ResNext50, self).__init__()
        self.model = models.resnext101_32x8d(pretrained = True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, nclasses)
    def forward(self, x):
        return self.model(x)
