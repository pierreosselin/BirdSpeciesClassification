from torchsummary import summary
import torch
from model import Net
from torchvision import datasets, models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 20)
model = model.to(device)
summary(model, (3, 64, 64))
