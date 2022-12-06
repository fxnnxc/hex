import torch 
import torch.nn as nn 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.c2 = nn.Sequential(
            nn.Conv2d(16,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),)

        self.c3 = nn.Sequential(
            nn.Conv2d(16,16,3,1,1),
            nn.ReLU(),
        )   
        self.final = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.c1(x) 
        x = self.c2(x)
        x = self.c3(x)
        x = self.final(x)
        return x
    