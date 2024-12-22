import torch
import torch.nn as nn

class ImprovedDeepBinaryClassifierNet(nn.Module):
    def __init__(self):
        super(ImprovedDeepBinaryClassifierNet, self).__init__()
        
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
        
        self.layer3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc1 = nn.Linear(512, 1)

        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc1(x).squeeze()

        # Apply sigmoid to output a probability value for binary classification
        output= self.sigmoid(x) 

        return output  
