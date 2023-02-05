import argparse
import os 
import torch 
import torch.nn.functional as F
from tqdm import tqdm 
import torchvision.transforms as T
import time 
import pickle 


parser = argparse.ArgumentParser()
parser.add_argument("--encoder")
parser.add_argument("--data-path", default='/data3/bumjin/bumjin_data/ILSVRC2012_val/')

args = parser.parse_args()
save_dir = f'results/{args.encoder}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- dataset 
name = args.encoder
if name == 'resnet50':
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
elif name == 'resnet18':
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
elif name == 'resnet34':
    from torchvision.models import resnet34, ResNet34_Weights
    weights = ResNet34_Weights.IMAGENET1K_V1
    model = resnet34(weights=weights)
elif name == 'resnet101':
    from torchvision.models import resnet101, ResNet101_Weights
    weights = ResNet101_Weights.IMAGENET1K_V1
    model = resnet101(weights=weights)
elif name == 'resnet152':
    from torchvision.models import resnet152, ResNet152_Weights
    weights = ResNet152_Weights.IMAGENET1K_V1
    model = resnet152(weights=weights)
elif name == 'inceptionv3':
    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights)
else:
    raise ValueError(f"{name} is not implemented")

# torch.save(model, f"{save_dir}/model.pt")
# --- Define Wrappers 

class BaseWrapper():
    def __init__(self, model):
        self.model = model
        
        self.fw_hooks = []
        self.fw_hook_modules = [] 
        self.classifier_modules = []
        self.classifier_hooks = []
        
        self.classifier_modules.append(self.model.fc)
        
        
    def _register_gap_stats_hooks(self):
        for conv in self.fw_hook_modules:        
            self.fw_hooks.append(conv.register_forward_hook(self.forward_hook_for_save_gap))
        
        for layer in self.classifier_modules:
            self.classifier_hooks.append(layer.register_forward_hook(self.fowward_hook_for_hidden_state))
        
    def fowward_hook_for_hidden_state(self, module, input, output):
        module.hidden = input[0].squeeze(0)
        
    def forward_hook_for_save_gap(self, module, input, output):
        output = F.relu(output)
        gap = F.adaptive_avg_pool2d(output, output_size=1)[0,:,0,0]
        temp = output.clone().detach()
        hw = output.size(-1) * output.size(-2)
        num_relu = (temp>0).view(gap.size(0), -1).sum(dim=-1)
        module.relu_ratio = num_relu/hw
        module.relu_position = input[0]>0
        module.gap = gap


class InceptionWrapper(BaseWrapper):
    def __init__(self, model):
        super().__init__(model)
        conv_modules = [] 
        conv_names = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3']
        conv_names += ['Mixed_5'+alpha for alpha in ['b','c','d']]
        conv_names += ['Mixed_6'+alpha for alpha in ['b','c','d','e']]
        conv_names += ['Mixed_7'+alpha for alpha in ['b','c']]
        for name in conv_names:
            if 'Mixed' in name:
                module_temp = getattr(model, name)
                module_temp = getattr(module_temp, 'branch_pool')
                conv_modules.append(getattr(module_temp, 'conv'))
            else:
                module_temp = getattr(model, name)
                conv_modules.append(getattr(module_temp, 'conv'))
        self.fw_hook_modules = conv_modules



class ResNetWrapper(BaseWrapper):
    def __init__(self, model):
        super().__init__(model)
        self.fw_hook_modules.append(self.model.conv1)
        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:        
            for basic_block in layer:
                for name in ['conv1', 'conv2', 'conv3']:
                    if hasattr(basic_block, name):
                        self.fw_hook_modules.append(getattr(basic_block, name))




from datasets import get_datasets, IMAGENET_MEAN, IMAGENET_STD
mean = IMAGENET_MEAN
std = IMAGENET_STD

transform = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean, std)
                ])

_, valid_dataset = get_datasets('imagenet1k', args.data_path, transform)

model.to("cuda:0")
model.eval()

if "resnet" in  args.encoder:
    wrapper = ResNetWrapper(model)
elif "inception" in args.encoder:
    wrapper = InceptionWrapper(model)
wrapper._register_gap_stats_hooks()

start_time = time.time()
pbar = tqdm(range(len(valid_dataset)))
tensors_RL = [[] for i in range(len(wrapper.fw_hook_modules))]   
tensors = [[] for i in range(len(wrapper.fw_hook_modules))]  
tensors_hidden = []
labels = []      
logits = []

for index in pbar:    
    x = valid_dataset[index][0].unsqueeze(0).to("cuda:0")
    acts = wrapper.model.forward(x)
    for i, module in enumerate(wrapper.fw_hook_modules):
        gap = module.gap.clone().detach().cpu()
        relu_ratio = module.relu_ratio.detach().clone().cpu()
        tensors[i].append(gap)
        tensors_RL[i].append(relu_ratio)
        
    for i, module in enumerate(wrapper.classifier_modules):
        hidden = module.hidden.clone().detach().cpu()
        tensors_hidden.append(hidden)
        
    labels.append(valid_dataset[index][1])
    logits.append(acts.squeeze(0).detach().cpu())
    duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))         
    pbar.set_description(f"🧪:[{save_dir}] E:({index/len(valid_dataset):.2f}) D:({duration})]")    
    
for i, module in enumerate(wrapper.fw_hook_modules):
    tensors[i] = torch.stack(tensors[i])
    tensors_RL[i] = torch.stack(tensors_RL[i])
    
# for t in tensors:
#     print(t.size())
with open(os.path.join(save_dir, f"valid_hidden.pkl"), 'wb') as f:
    pickle.dump(tensors_hidden, f, pickle.HIGHEST_PROTOCOL)   
with open(os.path.join(save_dir, f"valid_gap.pkl"), 'wb') as f:
    pickle.dump(tensors, f, pickle.HIGHEST_PROTOCOL)   
with open(os.path.join(save_dir, f"valid_labels.pkl"), 'wb') as f:
    pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)   
with open(os.path.join(save_dir, f"valid_logits.pkl"), 'wb') as f:
    pickle.dump(logits, f, pickle.HIGHEST_PROTOCOL)   
with open(os.path.join(save_dir, f"valid_relu_ratio.pkl"), 'wb') as f:
    pickle.dump(tensors_RL, f, pickle.HIGHEST_PROTOCOL)   
    

# save the statistics -------------
means = []
stds = [] 
for h in tensors:
    means.append(torch.mean(h, dim=0))
    stds.append(torch.std(h, dim=0))
import os 
with open(os.path.join(save_dir, f"valid_gap_mean.pkl"), 'wb') as f:
    pickle.dump(means, f, pickle.HIGHEST_PROTOCOL)   
with open(os.path.join(save_dir, f"valid_gap_std.pkl"), 'wb') as f:
    pickle.dump(stds, f, pickle.HIGHEST_PROTOCOL)   

# P-values for Welch's test -------------
