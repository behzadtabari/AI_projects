import torch.nn as nn


class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) model.

    Args:
        input_size (int): Number of input channels.
        output_size (int): Number of output units.
        batch_size (int): Batch size for training.
    """

    def __init__(self, input_size, output_size, batch_size):
        super(CNN, self).__init__()

        # First convolutional layer (kernel size ensures same padding)
        self.cnn1 = nn.Conv2d(in_channels=input_size, out_channels=8, kernel_size=3, padding=1, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(8)  # Batch normalization (8 channels)
        self.relu = nn.ReLU()  # ReLU activation function
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # Max pooling layer (2x2)

        # Second convolutional layer (kernel size ensures same padding)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, padding=2, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(32)  # Batch normalization (32 channels)

        # First fully connected layer (flattened input size: 32 * ((28/2)/2) * ((28/2)/2) = 1568)
        self.fc1 = nn.Linear(1568, 600)

        # Dropout layer (50% probability)
        self.dropout = nn.Dropout(p=0.5)

        # Output layer
        self.fc2 = nn.Linear(600, output_size)

        # Store batch size
        self.batch_size = batch_size

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """

        # Pass through first convolutional block
        x = self.cnn1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through second convolutional block
        x = self.cnn2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Flatten output for fully connected layers
        x = x.view(-1, 1568)

        # Fully connected layers with dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        return x
