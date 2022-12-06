import random
import torch 
from torch.autograd import Variable

def make_interpolation(x, base, M):
    lst = [] 
    for i in range(M+1):
        alpha = float(i/M)  
        interpolated =x * (alpha) + base * (1-alpha)
        lst.append(interpolated.clone())
    return torch.stack(lst)


def integrated_gradient(model, x, M, y=None, baseline=None, random_steps=0):
    device = x.device
    if random_steps > 0:
        return random_integrated_gradient_several(model, x, M, y, random_steps)
    
    if y is None:
        prediction_class = model(x.unsqueeze(0)).argmax(axis=-1).item()
        y = prediction_class
        
    baseline = baseline.to(device)
    X = make_interpolation(x, baseline, M)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    output = model.forward(X)
    score = torch.softmax(output, dim=-1)
    class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to("cuda")
    class_score[:,y] = score[:,y]
    output.backward(gradient=class_score)

    gradient = X.grad  #A pproximate the integral using the trapezoidal rule
    gradient = (gradient[:-1] + gradient[1:]) / 2.0

    output = (x - baseline) * gradient.mean(axis=0)
    return output

def random_integrated_gradient_several(model, x, M, y, random_steps):
    device = x.device
    all_grads = []
        
    if y is None:
        prediction_class = model(x.unsqueeze(0)).argmax(axis=-1).item()
        y = prediction_class
    for i in range(random_steps):
        baseline =torch.rand_like(x)

        X = make_interpolation(x, baseline, M)
        X = Variable(X, requires_grad=True).to(device)
        X.retain_grad()
        
        output = model.forward(X)
        score = torch.softmax(output, dim=-1)
        class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to("cuda")
        class_score[:,y] = score[:,y]

        class_score[:,prediction_class] = score[:,prediction_class]
        output.backward(gradient=class_score)

        gradient = X.grad  #A pproximate the integral using the trapezoidal rule
        gradient = (gradient[:-1] + gradient[1:]) / 2.0
        gradient =  (x - baseline) * gradient.mean(axis=0)
        all_grads.append(gradient)

    output= torch.mean(torch.stack(all_grads), axis=0)
    return output