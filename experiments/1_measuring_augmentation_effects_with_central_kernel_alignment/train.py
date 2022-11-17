

from data import get_cifar10_dataset
from model import Model
import torch 
from tqdm import tqdm 
import os 

trainset, testset, train_loader, valid_loader = get_cifar10_dataset()


if not os.path.exists("results"):
    os.makedirs("results")




for eps in [0, 0.1, 0.2, 0.3, 0.4]:
    model = Model().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.save(model, f"results/model_{eps}.pt")

    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(20):
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

        
        