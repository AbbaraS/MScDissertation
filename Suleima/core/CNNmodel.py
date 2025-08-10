
import torch
import torch.nn as nn
import torch.nn.functional as F



class SingleBranchCNN(nn.Module):
	def __init__(self, kernel_size=3,
				 padding=1,
				 dropout_rate=0.4):
		super(SingleBranchCNN, self).__init__()

		self.dropout_rate = dropout_rate
		self.kernel_size = kernel_size
		self.padding = padding

		self.conv1 = nn.Conv2d(3, 16, kernel_size=self.kernel_size, padding=self.padding)
		self.bn1 = nn.BatchNorm2d(16)
		self.pool1 = nn.MaxPool2d(2)
		# out1 ===>    (B, 16, H, W)      ===> input2
		self.conv2 = nn.Conv2d(16, 32,  kernel_size=self.kernel_size, padding=self.padding)
		self.bn2 = nn.BatchNorm2d(32)
		self.pool2 = nn.MaxPool2d(2)
		# out2 ===>    (B, 32, H, W)      ===> input3
		self.conv3 = nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.padding)
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
	H_out = H_in - 3 + 2*1 = H_in - 1
	H_in: 64  ===>   64 - 3 + 2*1 ===> H_out:  63
	H_in: 96  ===>   96 - 3 + 2*1 ===> H_out:  95
	H_in: 128 ===>  128 - 3 + 2*1 ===> H_out: 127
'''

class MultiViewCNN(nn.Module):
	def __init__(self,
				 use_metadata=True,
				 kernel_size=3,
				 padding=1,
				 dropout_rate=0.4):
		super(MultiViewCNN, self).__init__()

		self.use_metadata = use_metadata
		self.kernel_size = kernel_size
		self.padding = padding
		self.dropout_rate = dropout_rate

		self.axial_branch = SingleBranchCNN(self.kernel_size, self.padding, self.dropout_rate)
		self.sagittal_branch = SingleBranchCNN(self.kernel_size, self.padding, self.dropout_rate)
		self.coronal_branch = SingleBranchCNN(self.kernel_size, self.padding, self.dropout_rate)

		self.view_weights = nn.Parameter(torch.randn(3))

		total_features = 64 * 3
		if use_metadata:
			total_features += 2

		self.fc1 = nn.Linear(total_features, 128)
		self.dropout1 = nn.Dropout(dropout_rate)
		#self.dropout2 = nn.Dropout(dropout_rate)
		self.fc2 = nn.Linear(128, 1)  # Binary classification (output: logit)

	def forward(self, axial, sagittal, coronal, meta=None):
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

		if self.use_metadata and meta is not None:
			x = torch.cat([x, meta], dim=1)

		x = F.relu(self.fc1(x))
		x = self.dropout1(x)
		#x = self.dropout2(x)
		x = self.fc2(x)
		return x

