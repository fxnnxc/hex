import argparse
from data import get_cifar10_dataset
import torch 
from tqdm import tqdm 
import os 
import numpy as np 
import pickle
from model import Model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--init", type=str)

args = parser.parse_args()

save_path = f'results/{args.init}'

trainset, testset, train_loader, valid_loader = get_cifar10_dataset()
model_init =Model(lambda x:x).cuda()
model_init.load_state_dict(torch.load(f"{save_path}/model_init.pt"))
model_last =Model(lambda x:x).cuda()
model_last.load_state_dict(torch.load(f"{save_path}/model_last.pt"))


models = [model_init, model_last]
names = ['init', 'last']

def conv1_hook(module, input, output):
    global features
    features[0].append(output.clone().detach().cpu().numpy())

def conv2_hook(module, input, output):
    global features
    features[1].append(output.clone().detach().cpu().numpy())

def conv3_hook(module, input, output):
    global features
    features[2].append(output.clone().detach().cpu().numpy())

def conv1_before_relu_hook(module, input, output):
    global features
    features[3].append(output.clone().detach().cpu().numpy())

def conv2_before_relu_hook(module, input, output):
    global features
    features[4].append(output.clone().detach().cpu().numpy())

def conv3_before_relu_hook(module, input, output):
    global features
    features[5].append(output.clone().detach().cpu().numpy())

def final_hook(module, input, output):
    global features
    features[6].append(output.clone().detach().cpu().numpy())


for name, model in  zip(names, models):
    model.c1.register_forward_hook(conv1_hook)
    model.c2.register_forward_hook(conv2_hook)
    model.c3.register_forward_hook(conv3_hook)
    # ----
    model.c1[0].register_forward_hook(conv1_before_relu_hook)
    model.c2[0].register_forward_hook(conv2_before_relu_hook)
    model.c3[0].register_forward_hook(conv3_before_relu_hook)
    
    # ----
    model.final[1].register_forward_hook(final_hook) # before the logit
    
    # --- store train features
    pbar =  tqdm(train_loader)
    features = [[],[],[],[],[],[],[]]
    labels = []
    for i, (x,y) in enumerate(pbar):
        x = x.cuda()
        labels.append(y.cpu().numpy())
        y_hat = model(x)         

    for i in range(len(features)):
        features[i] = np.concatenate(features[i], axis=0)
    with open(os.path.join(save_path, f"{name}_train.pkl"), 'wb') as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
    # np.save(os.path.join(save_path, f"{name}_label.npy"), labels)

    # --- store valid features
    pbar =  tqdm(valid_loader)
    features = [[],[],[],[],[],[],[]]
    labels = []
    for i, (x,y) in enumerate(pbar):
        x = x.cuda()
        labels.append(y.cpu().numpy())
        y_hat = model(x)         
    
    for i in range(len(features)):
        features[i] = np.concatenate(features[i], axis=0)
    with open(os.path.join(save_path, f"{name}_valid.pkl"), 'wb') as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
    # np.save(os.path.join(save_path, f"{name}_label.npy"), labels)


    # --- store normal features
    pbar =  tqdm(valid_loader)
    features = [[],[],[],[],[],[],[]]
    labels = []
    for i, (x,y) in enumerate(pbar):
        x = x.cuda()
        x = torch.rand_like(x).cuda() # normal distribution
        y_hat = model(x)         

    for i in range(len(features)):
        features[i] = np.concatenate(features[i], axis=0)
    with open(os.path.join(save_path, f"{name}_random.pkl"), 'wb') as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)

