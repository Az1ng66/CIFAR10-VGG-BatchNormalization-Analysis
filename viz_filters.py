import torch
import matplotlib.pyplot as plt
from models.vgg import VGG_A_BatchNorm

model = VGG_A_BatchNorm()
model.load_state_dict(torch.load('best.pth'))
filters = model.features[0].weight.data.cpu()

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    f = filters[i].numpy().transpose(1, 2, 0)
    f = (f - f.min()) / (f.max() - f.min())
    ax.imshow(f)
    ax.axis('off')
plt.savefig('filters_viz.png')