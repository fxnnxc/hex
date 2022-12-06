import torchvision 
from torch.utils.data import DataLoader
import torchvision.transforms as T

def get_cifar10_dataset():
    root = "untracked"
    download = True 
    batch_size = 64
    num_workers = 1
    transform = T.Compose([T.Resize((32, 32)),
                                  T.ToTensor(),
                                  T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])

    data_train = torchvision.datasets.CIFAR10(root = root,  
                                    train = True,  
                                    transform = transform,
                                    download=download)
    data_test = torchvision.datasets.CIFAR10(root = root,
                                    train = False, 
                                    transform =transform,
                                    download=download)

    trainDataLoader = DataLoader(dataset = data_train,  
                                      batch_size = batch_size, 
                                      shuffle =True, 
                                      num_workers = num_workers) 

    testDataLoader = DataLoader(dataset = data_test, 
                                    batch_size = batch_size,
                                    shuffle = False, 
                                    num_workers = num_workers) 
                                    
    print ("[+] Finished loading data & Preprocessing")
    return  data_train, data_test, trainDataLoader,testDataLoader