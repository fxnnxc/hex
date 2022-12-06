

from data import get_cifar10_dataset
from model import Model
import torch 
from tqdm import tqdm 
import os 
if not os.path.exists("results"):
    os.makedirs("results")

# --------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--eps", default=0.0, type=float)

args = parser.parse_args()
eps = args.eps 
# --------

trainset, testset, train_loader, valid_loader = get_cifar10_dataset()
model = Model().cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
torch.save(model, f"results/model_{eps}.pt")

loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(100):
    running_loss = 0
    pbar =  tqdm(train_loader)
    for i, (x,y) in enumerate(pbar):
        x = x.cuda()
        x += (torch.rand_like(x)-0.5) * eps
        y = y.cuda()
        y_hat = model(x)         
        loss = loss_fn(y_hat, y)
        running_loss+= loss.item()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f"🔖 loss : {running_loss/(i+1)}")


torch.save(model, f"results/model_{eps}.pt")

    
    