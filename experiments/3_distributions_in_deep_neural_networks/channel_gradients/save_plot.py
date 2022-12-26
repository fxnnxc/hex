import torch 
import numpy 
import os 
import pickle
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--direction", type=str)
args = parser.parse_args()

model = f'resnet{args.model}'
order='bw'
direction= args.direction 

features = [] 
sample_path = os.path.join('results', model, f"{order}_{direction}")
layer_results_temp = {} # layer : [0~9999, 10000~19999, ...] 
for sample in tqdm(os.listdir(sample_path)):
    f = pickle.load(open(os.path.join(sample_path, sample), 'rb'))
    layer, sample_index = sample.split(".")[0].split("_") 
    if layer not in layer_results_temp.keys():
        layer_results_temp[layer] = [] 
    layer_results_temp[layer].append(f)

for k in layer_results_temp.keys():
    # print(np.concatenate(layer_results_temp[k], axis=0).shape)
    layer_results_temp[k] = np.concatenate(layer_results_temp[k], axis=0) # 50000 x Channels x Samples 

layer_results = layer_results_temp
layer_results.keys()

def plot_sns_mean_var(channel_rv, ax, **kwargs):
    channel_rv = channel_rv.transpose(1,0) # Channel x Sample x Dim 
    dic = {"channel":[],
           "value":[]}
    channel_rv = channel_rv[np.random.choice(range(channel_rv.shape[0]), 15, replace=False)]
    for i in range(channel_rv.shape[0]):
        cur_features = channel_rv[i]
        for v in cur_features:
            dic['channel'].append(i)
            dic['value'].append(v)
    df = pd.DataFrame(dic)
    del dic
    sns.boxplot(data=df, x='channel', y='value', ax=ax,  showfliers = False)

target_layers = layer_results.keys()
fig, axes = plt.subplots(1, len(target_layers), figsize=(len(target_layers)*2.6, 3))
axes_flat = axes.flat
fig.suptitle(f"{model}_{order}_{direction}_Var")
fig.supxlabel("Sorted Channel Index")
fig.supylabel("Var")

for j, target_layer in tqdm(enumerate(target_layers)): 
    channel_rv = layer_results[target_layer]
    ax = next(axes_flat)
    plot_sns_mean_var(channel_rv, ax, alpha=0.5,)
    ax.set_title(f'layer_{target_layer}')
    ax.set_ylabel("")
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f'results/{model}_{order}_{direction}_Var.pdf')