from core.CNNmodel import SingleBranchCNN
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewCNN(nn.Module):
    def __init__(self, input_size=(65, 65), 
                 use_metadata=False, 
                 ker_size=2, 
                 padd_size=1, 
                 dropout_rate=0.4):
        super(MultiViewCNN, self).__init__()

        self.use_metadata = use_metadata
        self.ker_size = ker_size
        self.padd_size = padd_size
        self.dropout_rate = dropout_rate
        H, W = input_size

        # === 1. Three convolutional branches for each view ===
        self.axial_branch = SingleBranchCNN(ker_size, padd_size, dropout_rate)
        self.sagittal_branch = SingleBranchCNN(ker_size, padd_size, dropout_rate)
        self.coronal_branch = SingleBranchCNN(ker_size, padd_size, dropout_rate)

        # === 2. Flattened size after two MaxPool2d(2) layers in each branch ===
        #flattened_size = 32 * (H // 4) * (W // 4)  # each branch output flattened

        # === 3. Learnable attention weights for 3 views ===
        # Initialized randomly, then optimized during training
        self.view_weights = nn.Parameter(torch.randn(3))  # shape: (3,)

        # === 4. Final total input size to the classifier ===
        #total_features = 3 * flattened_size       # all branches
        total_features = 32 * (H // 4) * (W // 4)  # each branch output flattened
        if use_metadata:
            total_features += 2  # e.g. age and gender

        # === 5. Classifier ===
        self.fc1 = nn.Linear(total_features, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)  # Binary classification (output: logit)

    def forward(self, axial, sagittal, coronal, meta=None):
        # === Step 1: Pass each view through its CNN branch ===
        a = self.axial_branch(axial)
        s = self.sagittal_branch(sagittal)
        c = self.coronal_branch(coronal)
        
        # === Step 2: Flatten each feature map ===
        a = a.view(a.size(0), -1)  # flatten
        s = s.view(s.size(0), -1)
        c = c.view(c.size(0), -1)
        #print(a.shape, s.shape, c.shape)
        # === Step 3: Stack into shape (B, 3, D) ===
        views = torch.stack([a, s, c], dim=1)  # (B, 3, D)
        
        # === Step 4: Normalize the attention weights using softmax ===
        weights = F.softmax(self.view_weights, dim=0)  # (3,)
        
        # === Step 5: Apply the weights to each view and sum ===
        # weights: (3,) → (1, 3, 1) → (B, 3, D)
        weighted_views = (views * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (B, D)
        
        # === Step 6: Concatenate metadata ===
        if self.use_metadata and meta is not None:
            x = torch.cat([weighted_views, meta], dim=1)
        else:
            x = weighted_views


        #x = torch.cat([a, s, c], dim=1)
        #if self.use_metadata and meta is not None:
        #    x = torch.cat([x, meta], dim=1)

        # === Step 7: Feedforward classifier ===
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # raw logit (no sigmoid)
        return x
    
