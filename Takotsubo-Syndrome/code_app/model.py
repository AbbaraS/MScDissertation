
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

# The same as in the training, but the hyperparams are set to the best values (obtained with Optuna). Also any code relating to training has been removed
class CNN3D(L.LightningModule):
    def __init__(self, learning_rate=0.00015489692592807923, dropout=0.3541126312412523, filters=(32, 64, 128), kernel_size=5, pooling_type="max", verbose=False):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.dropout_rate = dropout
        self.filters = filters
        self.kernel_size = kernel_size
        self.pooling_type = pooling_type
        self.verbose = verbose


        self.conv1 = nn.Conv3d(1, filters[0], kernel_size=self.kernel_size, stride=1, padding=1)
        if pooling_type == "max":
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        elif pooling_type == "avg":
            self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        else:
            raise ValueError("Invalid pooling type. Choose 'max' or 'avg'.")
        # self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(filters[0], filters[1], kernel_size=self.kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv3d(filters[1], filters[2], kernel_size=self.kernel_size, stride=1, padding=1)

        self.global_pool = nn.AdaptiveAvgPool3d((2, 2, 2))  # Output shape: (64, 2, 2, 2)

        # Fully connected layers for image features
        self.fc1 = nn.Linear(filters[2] * 2 * 2 * 2, 128)
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected layer for metadata input (Age + Gender)
        self.fc_metadata = nn.Linear(2, 16)  # 2 features: Age & Gender

        # Final classification layer (combining image + metadata)
        self.fc2 = nn.Linear(128 + 16, 1)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(91/76))  # Binary classification loss
        

    def forward(self, image, metadata):
        x = torch.relu(self.conv1(image))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))

        
        x = self.pool(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)


        # Process metadata separately
        meta_x = torch.relu(self.fc_metadata(metadata))

        # Concatenate image and metadata features
        x = torch.cat((x, meta_x), dim=1)

        # Final classification
        x = self.fc2(x)
        return x



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"} 