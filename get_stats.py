import torch
from models.vgg import VGG_A_BatchNorm

model = VGG_A_BatchNorm()
params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {params}")

