import torch
import torch.nn as nn

class ThreeD_CNN(nn.Module):
    """
    A simple 3D CNN model for video classification.
    """
    
    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): The number of output classes (e.g., number of verbs).
        """
        super().__init__()
        
        # This is the "feature extractor" part, made of 3D conv/pool blocks
        self.conv_layers = nn.Sequential(
            # Block 1
            # Input: [B, 3, T, H, W] (e.g., [8, 3, 16, 224, 224])
            # Kernel is 3x3x3 (Time, Height, Width)
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2), # Halves T, H, and W (e.g., -> [8, 16, 8, 112, 112])
            
            # Block 2
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2), # (e.g., -> [8, 32, 4, 56, 56])
            
            # Block 3
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)  # (e.g., -> [8, 64, 2, 28, 28])
        )
        
        # Flatten layer
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # This is the "classifier" part
        self.classifier = nn.Sequential(
            nn.Flatten(),           # Flattens [B, 64, 1, 1, 1] to [B, 64]
            nn.Linear(64, 32),      # The 64 matches the out_channels of the last conv
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        This defines the forward pass of the data through the model.
        
        Args:
            x (torch.Tensor): The input batch of video clips.
                              Shape: [B, C, T, H, W]
        """
        # 1. Pass through convolutional feature extractor
        x = self.conv_layers(x)
        
        # 2. Squash to a fixed-size feature vector
        x = self.adaptive_pool(x)
        
        # 3. Pass through the final classifier
        x = self.classifier(x)
        
        return x

