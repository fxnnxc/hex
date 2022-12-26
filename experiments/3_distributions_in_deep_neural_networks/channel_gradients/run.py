import pickle
import numpy as np 

def store_hook_result(arrays, save_name):
    # print("-----------")
    # print([t[0].squeeze(0).size() for t in tensors])
    arrays = np.stack(arrays)  # Layers x Channel x H x W 
    with open(save_name+'.pkl', 'wb') as f:
        pickle.dump(arrays, f)
    

# ----
# Model 
def get_encoder(name):
    if name == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        return model 
    elif name == 'resnet18':
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        return model 
    elif name == 'resnet34':
        from torchvision.models import resnet34, ResNet34_Weights
        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
        return model     
    elif name == 'resnet101':
        from torchvision.models import resnet101, ResNet101_Weights
        weights = ResNet101_Weights.DEFAULT
        model = resnet101(weights=weights)
        return model     
    elif name == 'resnet152':
        from torchvision.models import resnet152, ResNet152_Weights
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights)
        return model     
    else:
        raise ValueError(f"{flags.encoder} is not implemented")

import argparse
from omegaconf import OmegaConf
import torchvision
from torchvision import transforms
import json 
import os 
from tqdm import tqdm 
from hook_wrapper import ResNetHookHelper
import torch
from distutils.util import strtobool


parser = argparse.ArgumentParser()
parser.add_argument("--encoder", type=str)
parser.add_argument("--target-layer", type=int)
parser.add_argument("--num-channels", type=int)
parser.add_argument("--num-flat-samples", type=int)

args = parser.parse_args()
flags = OmegaConf.create({})
for key in vars(args):
    setattr(flags, key, getattr(args, key))

flags.data_root = '/home/bumjin/data/ILSVRC2012_val'


model = get_encoder(flags.encoder).to("cuda:0").train()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225] 
preprocess = transforms.Compose([
                            # transforms.ToPILImage(),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std),    
                        ])
wrapper = ResNetHookHelper(model, flags.target_layer, flags.num_channels, flags.num_flat_samples)

path = flags.data_root 
testset = torchvision.datasets.ImageNet(root=path, split='val')
with open(os.path.join(path, "imagenet_label.json"), "rb") as f:
    labels = json.load(f)

save_path = f'results/{flags.encoder}'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    # os.makedirs(os.path.join(save_path, 'fw_in'))
    # os.makedirs(os.path.join(save_path, 'fw_out'))
    os.makedirs(os.path.join(save_path, 'bw_in'))
    os.makedirs(os.path.join(save_path, 'bw_out'))

from torch.autograd import Variable 

count = 0 
for index in tqdm(range(len(testset))):
    x,y = testset[index] 
    x = preprocess(x).unsqueeze(0).to("cuda:0")
    y = torch.tensor(y).unsqueeze(0).to("cuda:0")
    
    x = x.to("cuda:0")
    x = Variable(x, requires_grad=True).to("cuda:0")
    y = y.to("cuda:0")
    wrapper.zero_grad()
    
    output = wrapper.forward(x)
    score = torch.softmax(output, dim=-1)
    class_score = torch.FloatTensor(x.size(0), output.size()[-1]).zero_().to("cuda")
    class_index = y
    class_score[:,class_index] = score[:,class_index]
    output.backward(gradient=class_score)

    if index %10000 == 9999:
        # store_hook_result(wrapper.fw_in_holder,  os.path.join(save_path, 'fw_in',  f'{flags.target_layer}'))
        # store_hook_result(wrapper.fw_out_holder, os.path.join(save_path, 'fw_out', f'{flags.target_layer}'))
        store_hook_result(wrapper.bw_in_holder,  os.path.join(save_path, 'bw_in',  f'{flags.target_layer}_{index}'))
        store_hook_result(wrapper.bw_out_holder, os.path.join(save_path, 'bw_out', f'{flags.target_layer}_{index}'))
        wrapper.clear_holder()

# ---save 

# ---remove 