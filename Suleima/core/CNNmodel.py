import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)


class SingleBranchCNN(nn.Module):
	def __init__(self, dropout_rate=0.2):
		super().__init__()

		self.dropout_rate = dropout_rate

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(16)
		self.pool1 = nn.MaxPool2d(2)
		# out1 ===>    (B, 16, H, W)      ===> input2
		self.conv2 = nn.Conv2d(16, 32,  kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.pool2 = nn.MaxPool2d(2)
		# out2 ===>    (B, 32, H, W)      ===> input3
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.pool3 = nn.MaxPool2d(2)
		# out3 ===>    (B, 64, H, W)      ===> output
		self.dropout = nn.Dropout2d(self.dropout_rate)
		self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))   # Output shape: (B, 64, 1, 1)

	def forward(self, x):
		x = self.pool1(F.relu(self.bn1(self.conv1(x))))
		x = self.pool2(F.relu(self.bn2(self.conv2(x))))
		x = self.pool3(F.relu(self.bn3(self.conv3(x))))
		x = self.dropout(x)
		x = self.adaptive_pool(x)
		return x

'''
H_out = (H_in - kernel_size + (2×padding) / stride) + 1

H_out = H_in - 3 + 2*1 + 1
This means:
	H_in: 32  ===>   32 - 3 + 2*1 ===> H_out:  32
	H_in: 64  ===>   64 - 3 + 2*1 ===> H_out:  64
	H_in: 96  ===>   96 - 3 + 2*1 ===> H_out:  96
	H_in: 128 ===>  128 - 3 + 2*1 ===> H_out: 128
'''

class MultiViewCNN(nn.Module):
	def __init__(self, dropout_rate=0.2):
		super().__init__()

		self.axial_branch = SingleBranchCNN(dropout_rate)          # 64 features
		self.sagittal_branch = SingleBranchCNN(dropout_rate)       # 64 features
		self.coronal_branch = SingleBranchCNN(dropout_rate)        # 64 features
		self.view_weights = nn.Parameter(torch.randn(3))

		total_features = (64 * 3) + 3
		self.fc1 = nn.Linear(total_features, 128)
		self.dropout1 = nn.Dropout(dropout_rate)
		self.fc2 = nn.Linear(128, 1)  # Binary classification (output: logit)

	def forward(self, axial, sagittal, coronal, meta):
		a = self.axial_branch(axial)
		s = self.sagittal_branch(sagittal)
		c = self.coronal_branch(coronal)

		a = a.view(a.size(0), -1) # flattens (B, 64, 1, 1) to (B, 64)
		s = s.view(s.size(0), -1)
		c = c.view(c.size(0), -1)

		weights = F.softmax(self.view_weights, dim=0)
		weighted_a = a * weights[0]
		weighted_s = s * weights[1]
		weighted_c = c * weights[2]

		x = torch.cat([weighted_a, weighted_s, weighted_c], dim=1) # shape: (B, D*3)
		x = torch.cat([x, meta], dim=1)

		x = F.relu(self.fc1(x))
		x = self.dropout1(x)
		x = self.fc2(x)
		return x



class SingleViewClassifier(nn.Module):
	def __init__(self, dropout_rate=0.4):
		super(SingleViewClassifier, self).__init__()
		self.view_features = SingleBranchCNN(dropout_rate)

		total_features = 64  + 3

		self.fc1 = nn.Linear(total_features, 64)
		self.dropout = nn.Dropout(dropout_rate)
		self.fc2 = nn.Linear(64, 1)

	def forward(self, image, meta):

		view_features = self.view_features(image)
		view_features = view_features.view(view_features.size(0), -1)

		combined_features = torch.cat([view_features, meta], dim=1)

		x = F.relu(self.fc1(combined_features))
		x = self.dropout(x)
		x = self.fc2(x)
		return x