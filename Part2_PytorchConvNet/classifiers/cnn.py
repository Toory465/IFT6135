import torch
import torch.nn as nn


# Create model
class ConvNet(nn.Module):
    """Convnet Classifier"""
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # Convolutional Layers
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            # Layer 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),

            # Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            # Layer 4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        
        # Fully Connected layers
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x).view(x.shape[0], -1))