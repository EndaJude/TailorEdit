import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalGate(nn.Module):
    def __init__(self, input_channel=77, feature_dim = 768, output_dim=4):
        super(GlobalGate, self).__init__()
        
        self.fc1 = nn.Linear(input_channel * feature_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)