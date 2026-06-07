import torch
import torch.nn as nn
import torch.optim as optim
import time
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loader = get_cifar_loader(root='./data', batch_size=128, train=True, num_workers=0)
    test_loader = get_cifar_loader(root='./data', batch_size=128, train=False, num_workers=0)

    model = VGG_A_BatchNorm().to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.StepLR(opt, 15, 0.1)

    for ep in range(20):
        model.train()
        start = time.time()
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
            
            if i % 10 == 0:
                print(f"Epoch {ep+1} | Batch {i}/{len(loader)} | Loss: {crit(model(x),y).item():.4f}", end='\r')
        
        model.eval()
        ok, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                _, p = torch.max(model(x), 1)
                total += y.size(0)
                ok += (p == y).sum().item()
        
        print(f"\nEpoch {ep+1} Finished | Acc: {100*ok/total:.2f}% | Time: {time.time()-start:.1f}s")
        sched.step()
    
    torch.save(model.state_dict(), 'best.pth')

if __name__ == '__main__':
    train()