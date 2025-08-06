import torch.nn as nn
import torch.nn.functional as F


class SingleBranchCNN(nn.Module):
    def __init__(self, ker_size=2, 
                 padd_size=1, 
                 dropout_rate=0.4):
        super(SingleBranchCNN, self).__init__()

        self.dropout_rate = dropout_rate
        self.ker_size = ker_size
        self.padd_size = padd_size
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=ker_size, padding=padd_size)  
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)                                        # (B, 16, H/2, W/2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=ker_size, padding=padd_size)            # (B, 32, H/2, W/2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)                                        # (B, 32, H/4, W/4)
        
        self.dropout = nn.Dropout2d(dropout_rate)  # Dropout for feature maps

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        return x  # Output shape: (B, 32, H/4, W/4)
    