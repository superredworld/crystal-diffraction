import torchsummary
import torchvision.models
import torch
# import torchsummarych
# from src import UNet
from model import convnext_tiny

num_classes = 3
model = convnext_tiny(num_classes=num_classes)

device = torch.device('cuda')
model.to(device)

torchsummary.summary(model.cuda(), (3, 224, 224))
# torchsummary.summary(UNet.cuda(), (3, 512, 512))