import pickle 
import torch 
import os 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import json 
from datasets import get_datasets
import torchvision.transforms as T 
import argparse
import time 

parser = argparse.ArgumentParser()
parser.add_argument("--encoder")
parser.add_argument("--data-path", default='/data3/bumjin/bumjin_data/ILSVRC2012_val/')
parser.add_argument("--top-k", type=int)

args = parser.parse_args()
save_dir = f'results/{args.encoder}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- dataset 
name = args.encoder

from datasets import get_datasets, IMAGENET_MEAN, IMAGENET_STD
mean = IMAGENET_MEAN
std = IMAGENET_STD

logits =  pickle.load(open(f"results/{args.encoder}/valid_logits.pkl", 'rb'))
logits = torch.stack(logits).to("cuda:0")

def get_top_k_indices(cls, indices, top_k=5):
    return  indices[cls, :top_k]
            

sims = pickle.load(open(f"results/{args.encoder}/sims.pkl", 'rb'))
indices = {}
for k,v in sims.items():
    v =  v.to("cuda:0")
    v, idx = torch.sort(v, descending=True, dim=1)
    indices[k] = idx

start_time = time.time()
TOP_K = args.top_k

cls_eqs = {} 


for sim_name in ['gap_pure', 'gap_stats', 'hidden']:
    cls_eqs[sim_name] = [0 for i in range(1000)]
    for cls in tqdm(range(1000)):
        top_k_indices = get_top_k_indices(cls, indices[sim_name], top_k=TOP_K)        
        
        X = torch.stack([50*top_k_indices[i]+j for i in range(len(top_k_indices)) for j in range(50) ]).to('cuda:0')
        Y = torch.stack([top_k_indices[i] for i in range(len(top_k_indices)) for j in range(50) ]).to('cuda:0')

        output = logits[X,:].clone()
        output[:, ~top_k_indices] = -torch.inf
        y_hat = output.argmax(dim=1)
        cls_eqs[sim_name][cls] = (y_hat == Y).sum().item() / len(Y)
        
with open(f"{save_dir}/cls_eqs_{TOP_K}.pkl", 'wb') as f:
    pickle.dump(cls_eqs, f, pickle.HIGHEST_PROTOCOL)
        
        
    
    
    # get top_k labels 
    


# for index in pbar:    
#     x = valid_dataset[index][0].unsqueeze(0).to("cuda:0")
#     outputs = model.forward(x)

#     logits.append(outputs.squeeze(0).detach().cpu())
#     duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))         
#     pbar.set_description(f"🧪:[{save_dir}] E:({index/len(valid_dataset):.2f}) D:({duration})]")    
    