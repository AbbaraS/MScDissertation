import torch
import torch.nn as nn
from torchvision import models

def create_adapted_resnet18(device):
	"""
	Creates a pre-trained ResNet-18 model adapted for binary classification.

	Args:
		device: The torch device to move the model to ('cuda' or 'cpu').

	Returns:
		The modified model and a list of parameters to be updated.
	"""
	# 1. Load a pre-trained ResNet-18 model
	# Use the recommended 'weights' argument for modern PyTorch
	print("Loading pre-trained ResNet-18 model...")
	model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

	# 2. Freeze all the parameters in the model
	# This prevents the pre-trained weights from changing during training
	print("Freezing existing model layers...")
	for param in model.parameters():
		param.requires_grad = False

	# 3. Replace the final fully connected layer (the 'classifier')
	# Get the number of input features for the final layer
	num_ftrs = model.fc.in_features

	# Create a new final layer for binary classification (outputs a single logit)
	# The new layer's parameters will have requires_grad=True by default
	model.fc = nn.Linear(num_ftrs, 1)
	print(f"Replaced final layer. New layer: {model.fc}")

	# Move the model to the specified device (GPU or CPU)
	model = model.to(device)

	# Collect the parameters of the new layer that need to be trained
	params_to_update = [param for param in model.parameters() if param.requires_grad]

	return model, params_to_update