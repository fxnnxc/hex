from data import get_cifar10_dataset
from model import Model
import torch 
from tqdm import tqdm 
import os 
import numpy as np 
from torch.utils.tensorboard import SummaryWriter 

 
# -------------------------------------------


def xavier_uniform_init(layer):
    torch.nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))
    torch.nn.init.constant_(layer.bias, 0)
    return layer

def xavier_normal_init(layer):
    torch.nn.init.xavier_normal_(layer.weight, gain=np.sqrt(2))
    torch.nn.init.constant_(layer.bias, 0)
    return layer

def kaiming_normal_init(layer):
    torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    torch.nn.init.constant_(layer.bias, 0)
    return layer

def kaiming_uniform_init(layer):
    torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    torch.nn.init.constant_(layer.bias, 0)
    return layer

def zeros_init(layer):
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.constant_(layer.bias, 0)
    return layer

def ones_init(layer):
    torch.nn.init.ones_(layer.weight)
    torch.nn.init.constant_(layer.bias, 0)
    return layer

def orthogonal_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, 0)
    return layer

# -------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--init", type=str)

args = parser.parse_args()

init_fn = {
    'xavier_normal' : xavier_normal_init,
    'xavier_uniform' : xavier_uniform_init,
    'kaiming_normal' : kaiming_normal_init,
    'kaiming_uniform' : kaiming_uniform_init,
    'ones' : ones_init,
    'zeros' : zeros_init,
    'orthogonal' : orthogonal_init,    
}[args.init]
# --------
save_path = f'results/{args.init}'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    

# --------
trainset, testset, train_loader, valid_loader = get_cifar10_dataset()
model = Model(init_fn).cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
torch.save(model.state_dict(), f"{save_path}/model_init.pt")
writer = SummaryWriter(save_path)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(50):
    running_loss = 0
    pbar =  tqdm(train_loader)
    for i, (x,y) in enumerate(pbar):
        x = x.cuda()
        y = y.cuda()
        y_hat = model(x)         
        loss = loss_fn(y_hat, y)
        running_loss+= loss.item()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f"🔖 loss : {running_loss/(i+1)}")
    writer.add_scalar("loss", running_loss/(i+1), epoch)

torch.save(model.state_dict(), f"{save_path}/model_last.pt")

    
    