import torch 
from torch.autograd import Variable

def make_perturbation(x, M, sigma=1):
    lst = [] 
    for i in range(M):
        noise = torch.normal(0, sigma, size=x.size()).to(x.device)
        lst.append(x.clone() + noise.clone())
    return torch.stack(lst)


def smooth_gradient(model, x, M, sigma, y=None):
    
    device = x.device
    sigma = sigma * (x.max() - x.min())
    X = make_perturbation(x, M, sigma)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    output = model.forward(X)
    score = torch.softmax(output, dim=-1)
    class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to(device)
    if y is None:
        class_index = output.argmax(axis=-1)
    else:
        class_index = y
    class_score[:,class_index] = score[:,class_index]
    output.backward(gradient=class_score)

    gradient = X.grad #* X.grad  # magnitude
    output = gradient.sum(0).detach()
    return output