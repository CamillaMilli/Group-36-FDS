import torch
import torch.nn as nn

class BinaryClassifierNet(nn.Module):
    def __init__(self):
        super(BinaryClassifierNet, self).__init__()    
        
        self.layer1 = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),  
            nn.ReLU()  
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),  
            nn.ReLU()
        )
        
        self.fc = nn.Linear(128, 1)  
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x).squeeze()
        # Apply sigmoid to output a probability value for binary classification
        output= self.sigmoid(x)   
        return output
