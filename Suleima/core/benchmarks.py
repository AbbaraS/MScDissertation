import torch
import torch.nn as nn
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.CNNmodel import *


class MetadataMLP(nn.Module):
	def __init__(self, input_features=3, dropout_rate=0.4): # Input: age & gender
		super(MetadataMLP, self).__init__()
		self.fc1 = nn.Linear(input_features, 64)
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

class SingleViewClassifier(nn.Module):
	def __init__(self, dropout_rate=0.4):
		super(SingleViewClassifier, self).__init__()
		# Use one instance of your existing single-branch CNN
		self.view_features = SingleBranchCNN(dropout_rate)

		total_features = 64  + 3
		# Classifier head
		# The feature_extractor outputs a (B, 64, 1, 1) tensor, which flattens to 64 features
		self.fc1 = nn.Linear(total_features, 64) # 64 image features + 2 meta features
		self.dropout = nn.Dropout(dropout_rate)
		self.fc2 = nn.Linear(64, 1)

	def forward(self, image, meta):
		# We'll test this with the axial view, but you can run it for each view
		view_features = self.view_features(image)
		view_features = view_features.view(view_features.size(0), -1) # Flatten

		# Combine image and metadata features
		combined_features = torch.cat([view_features, meta], dim=1)

		x = F.relu(self.fc1(combined_features))
		x = self.dropout(x)
		x = self.fc2(x)
		return x


# In training loop for this model:
# feature_extractor, classifier = create_adapted_resnet18(device)
# image_features = feature_extractor(axial_images) # Pass one view
# combined_input = torch.cat([image_features, meta], dim=1)
# outputs = classifier(combined_input)

def create_adapted_resnet18(device):
	"""
	Creates a pre-trained ResNet-18 model adapted for binary classification.
	"""
	# Load a pre-trained ResNet-18 model
	feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

	# Freeze all the parameters in the model
	# This prevents the pre-trained weights from changing during training
	print("Freezing existing model layers...")
	for param in feature_extractor.parameters():
		param.requires_grad = False


	# Get the number of input features for the final layer
	num_ftrs = feature_extractor.fc.in_features    ## This will be 512

	feature_extractor.fc = nn.Identity()

	classifier = nn.Sequential(
		nn.BatchNorm1d(num_ftrs + 3), # 512 resnet features + 2 meta features
		nn.Linear(num_ftrs + 2, 256),
		nn.ReLU(),
		nn.Dropout(0.5),
		nn.Linear(256, 1)
	)
	feature_extractor = feature_extractor.to(device)
	classifier = classifier.to(device)
	return feature_extractor.to(device), classifier.to(device)