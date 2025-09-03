import torch
import torch.nn as nn
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F



class MetadataMLP(nn.Module):
	def __init__(self, dropout_rate=0.4): # Input: age & gender
		super(MetadataMLP, self).__init__()
		self.fc1 = nn.Linear(3, 64)
		self.bn1 = nn.BatchNorm1d(64)
		self.dropout1 = nn.Dropout(dropout_rate)
		self.fc2 = nn.Linear(64, 32)
		self.bn2 = nn.BatchNorm1d(32)
		self.dropout2 = nn.Dropout(dropout_rate)
		self.fc3 = nn.Linear(32, 1) # Binary classification output

	def forward(self, meta):
		# Assuming 'meta' is your tensor of [age, gender]
		x = F.relu(self.bn1(self.fc1(meta)))
		x = self.dropout1(x)
		x = F.relu(self.bn2(self.fc2(x)))
		x = self.dropout2(x)
		x = self.fc3(x)
		return x




def create_adapted_resnet18(device):
	"""
	Creates a pre-trained ResNet-18 model adapted for binary classification.
	"""
	# Load a pre-trained ResNet-18 model
	m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
	W = m.conv1.weight.data.mean(dim=1, keepdim=True)  # [64,1,7,7]
	m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
	m.conv1.weight.data = W

	# Freeze all the parameters in the model
	# This prevents the pre-trained weights from changing during training
	#print("Freezing existing model layers...")
	for p in m.parameters(): p.requires_grad = False
	for p in m.layer4.parameters(): p.requires_grad = True  # optionally layer3 too
	optimizer = torch.optim.Adam([
    {"params": m.layer4.parameters(), "lr": 1e-4},
    {"params": classifier.parameters(), "lr": 5e-4}], weight_decay=1e-4)

	# Get the number of input features for the final layer
	num_ftrs = m.fc.in_features    ## This will be 512
	m.fc = nn.Identity()
	num_views = 3
	total_input_features = (num_ftrs * num_views) + 3     # (512 * 3) + 3 = 1539

	classifier = nn.Sequential(
		nn.LayerNorm(total_input_features), # 512 resnet features + 2 meta features
		nn.Linear(total_input_features, 256),
		nn.ReLU(),
		nn.Dropout(0.3),
		nn.Linear(256, 1)
	)

	return m.to(device), classifier.to(device)

