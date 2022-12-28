
import torch 
import torch.nn.functional as F
import numpy as np 

def build_backward_conv2d_hook(channels, in_holder, out_holder, emission_holder):
    def backward_conv2d_hook(module, grad_inputs, grad_outputs):
        weight, bias = module.weight, module.bias
        # ---
        new_weight = weight.clone()
        new_grad_inputs = torch.nn.grad.conv2d_input(grad_inputs[0].shape, new_weight, grad_outputs[0],
                            stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)

        # emission 
        weight_temp = weight.clone()
        ems = []
        MAX_SAMPLES = 15
        for k in range(min(weight_temp.size(0), MAX_SAMPLES)):
            channel_weight = weight_temp[k,...].unsqueeze(0).clone()
            channel_grads = grad_outputs[0][:,k,...].unsqueeze(1).clone()
            emitted_grads = torch.nn.grad.conv2d_input(grad_inputs[0].shape, channel_weight, channel_grads,
                            stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
            em = emitted_grads.norm()
            # print(emitted_grads.size())
            ems.append(em.item())
            
        grads_in = grad_inputs[0][0][:MAX_SAMPLES] # do it for the first 15 channels
        grads_in = grads_in.reshape(grads_in.size(0), -1).norm(dim=-1)

        grads_out = grad_outputs[0][0][:MAX_SAMPLES] # do it for the first 15 channels
        grads_out = grads_out.reshape(grads_out.size(0), -1).norm(dim=-1)

        in_holder.append(grads_in.cpu().numpy())
        out_holder.append(grads_out.cpu().numpy())
        emission_holder.append(np.array(ems))
        
        # out_holder.append(grad_outputs[0].cpu())
        return [new_grad_inputs]
    return backward_conv2d_hook

def build_forward_hook(in_holder, out_holder):
    def forward_hook(module, input, output):
        pass
        # in_holder.append(input[0].cpu())
        # out_holder.append(output[0].cpu())
    return forward_hook

class ResNetHookHelper():
    def __init__ (self, model, target_layer, num_channels):
        self.model = model
        # self.fw_hooks = [] 
        self.bw_hooks = [] 

        # self.fw_in_holder = []
        # self.fw_out_holder = []
        
        self.emission_holder = []
        self.bw_in_holder = []
        self.bw_out_holder = []
        
        self.num_channels = num_channels
        self.num_layers = self.compute_number_of_hook_layers()
        self.target_layer = target_layer
        
        
        print(f"👉 total layers: {self.num_layers} : your choice :{target_layer}")
        assert self.target_layer < self.num_layers
        self._register_hook()
    
    def compute_channel_indices(self):
        self.random_indices = np.random.choice(range(self.num_target_layer_channel),
                                               min(self.num_target_layer_channel, self.num_channels), 
                                               replace=False)    
    
    def compute_number_of_hook_layers(self):
        count = 1 
        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:        
            for basic_block in layer:
                for name in ['conv1', 'conv2', 'conv3']:
                    if hasattr(basic_block, name):
                        count+=1 
        return count

    def _register_hook(self):
        count = 0
        if self.target_layer ==0:
            # self.fw_hooks.append(self.model.conv1.register_forward_hook(build_forward_hook(self.fw_in_holder, self.fw_out_holder)))
            self.num_target_layer_channel = self.model.conv1.weight.size(1)
            self.compute_channel_indices()
            self.bw_hooks.append(self.model.conv1.register_full_backward_hook(build_backward_conv2d_hook(self.random_indices, self.bw_in_holder, self.bw_out_holder, self.emission_holder)))
            return 
        
        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:        
            for basic_block in layer:
                for name in ['conv1', 'conv2', 'conv3']:
                    if hasattr(basic_block, name):
                        count +=1 
                        if self.target_layer == count:
                            self.num_target_layer_channel = getattr(basic_block, name).weight.size(1)
                            self.compute_channel_indices()  
                            # self.fw_hooks.append(getattr(basic_block, name).register_forward_hook(build_forward_hook(self.fw_in_holder, self.fw_out_holder)))
                            self.bw_hooks.append(getattr(basic_block, name).register_full_backward_hook(build_backward_conv2d_hook(self.random_indices, self.bw_in_holder, self.bw_out_holder, self.emission_holder)))
                            return 
                        
    def _remove_hook(self):
        while len(self.bw_hooks):
            self.bw_hooks.pop().remove()
        # while len(self.fw_hooks):
        #     self.fw_hooks.pop().remove()
               
    def clear_holder(self):
        # self.fw_in_holder.clear()
        # self.fw_out_holder.clear()
        self.bw_in_holder.clear()
        self.bw_out_holder.clear()
        self.emission_holder.clear()
    
    def forward(self, x):    
        output = self.model(x)
        # self._remove_hook()
        return output

    def zero_grad(self):
        self.model.zero_grad()
        