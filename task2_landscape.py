import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

def run_experiment(model_class, device, lrs, steps, bs):
    results = []
    for lr in lrs:
        print(f"Testing {model_class.__name__} | LR: {lr}")
        model = model_class().to(device)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        loader = get_cifar_loader(root='./data', batch_size=bs, train=True, num_workers=0)
        
        losses = []
        count = 0
        while count < steps:
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                count += 1
                if count % 10 == 0: print(f"Step: {count}/{steps}", end='\r')
                if count >= steps: break
        results.append(losses)
    res = np.array(results)
    return np.min(res, axis=0), np.max(res, axis=0), np.mean(res, axis=0)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lrs, steps, bs = [1e-3, 2e-3, 1e-4, 5e-4], 150, 128

    v_min, v_max, v_avg = run_experiment(VGG_A, device, lrs, steps, bs)
    b_min, b_max, b_avg = run_experiment(VGG_A_BatchNorm, device, lrs, steps, bs)

    plt.figure(figsize=(10, 6))
    s = np.arange(steps)
    plt.fill_between(s, v_min, v_max, color='g', alpha=0.2, label='VGG Range')
    plt.plot(s, v_avg, 'g', label='VGG Mean')
    plt.fill_between(s, b_min, b_max, color='r', alpha=0.2, label='VGG+BN Range')
    plt.plot(s, b_avg, 'r', label='VGG+BN Mean')
    plt.legend(); plt.grid(True)
    plt.savefig('landscape.png')