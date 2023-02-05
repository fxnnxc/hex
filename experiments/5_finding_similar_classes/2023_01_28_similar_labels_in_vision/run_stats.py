import argparse
import os 
import torch 
from tqdm import tqdm 
import pickle 
import scipy.stats as stats

parser = argparse.ArgumentParser()
parser.add_argument("--encoder")
parser.add_argument("--data-path", default='/data3/bumjin/bumjin_data/ILSVRC2012_val/')

args = parser.parse_args()

encoder = args.encoder
save_dir = f"results/{encoder}"

tensors = pickle.load(open(os.path.join(save_dir, f"valid_gap.pkl"), 'rb'))

labels = torch.tensor(pickle.load(open(f"results/{encoder}/valid_labels.pkl", 'rb')))
MAX_CLS = max(labels)
cls_statistics = [torch.zeros(p.size(1), MAX_CLS+1) for p in tensors]

for cls in tqdm(range(MAX_CLS+1)):
    cls_labels = labels==cls 
    non_cls_labels = ~cls_labels
    for i in range(len(tensors)):
        h = tensors[i]
        for c in range(h.size(1)):
            population_cls = h[cls_labels, c]
            population_remaining = h[non_cls_labels, c]
            # use 
            statistics = stats.ttest_ind(population_cls, population_remaining,  alternative='greater', equal_var=False)[0]
            cls_statistics[i][c,cls] = statistics

with open(os.path.join(save_dir, f"cls_statistics.pkl"), 'wb') as f:
    pickle.dump(cls_statistics, f, pickle.HIGHEST_PROTOCOL)
    