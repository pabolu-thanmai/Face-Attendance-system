import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhanceNetNoPool(nn.Module):
    def __init__(self):
        super(EnhanceNetNoPool, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        # Convolutional Layers
        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(32, 24, 3, 1, 1, bias=True)
    
    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(x4))
        x6 = self.relu(self.e_conv6(x5))
        x_r = torch.tanh(self.e_conv7(x6))
        
        x_r1, x_r2, x_r3, x_r4, x_r5, x_r6, x_r7, x_r8 = torch.split(x_r, 3, dim=1)
        
        x = x + x_r1 * (torch.pow(x, 2) - x)
        x = x + x_r2 * (torch.pow(x, 2) - x)
        x = x + x_r3 * (torch.pow(x, 2) - x)
        x = x + x_r4 * (torch.pow(x, 2) - x)
        x = x + x_r5 * (torch.pow(x, 2) - x)
        x = x + x_r6 * (torch.pow(x, 2) - x)
        x = x + x_r7 * (torch.pow(x, 2) - x)
        enhance_image = x + x_r8 * (torch.pow(x, 2) - x)

        return enhance_image

def enhance_net_nopool():
    return EnhanceNetNoPool()
