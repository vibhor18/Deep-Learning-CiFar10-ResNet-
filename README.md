# Deep-Learning-CiFar10-ResNet-
 CS-GY 6953 : Deep Learning Mini Project


## To reproduce results:
1. Download best_model3.pth from the repository
2. Run this block of code
   ```
   import torchvision
   from torchsummary import summary
   import torch
   import torchvision
   import torchvision.transforms as transforms
   import torch.nn as nn
   from torch import optim
   import torch.utils.data as data
   import numpy as np


   class Resnet_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # Call to the constructor of the parent class (nn.Module)
        super(Resnet_block, self).__init__()

        # First convolution layer with bias disabled to work with BatchNorm
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # Batch normalization layer to stabilize and accelerate training
        self.bn1 = nn.BatchNorm2d(out_channels)
        # ReLU activation function to introduce non-linearity
        self.relu1 = nn.ReLU()

        # Second convolution layer with bias disabled as first
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        # Second batch normalization layer following the second convolution
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Second ReLU activation function
        self.relu2 = nn.ReLU()

        # Initialize the residual path as an empty sequential container
        self.residual = nn.Sequential()
        # Adjust the residual path if stride is not 1 or channel sizes change
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                # Convolution layer to adjust dimensions and stride in residual path
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                # Batch normalization layer for the adjustment layer
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Forward pass through the first convolutional, batch norm, and ReLU layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Forward pass through the second convolutional, batch norm layers
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the output from the residual path to the main path
        out += self.residual(x)
        # Final ReLU activation after merging the paths
        out = self.relu2(out)

        return out

   class custom_Resnet(nn.Module):
    def __init__(self, block, n_start_filters, layers, num_classes, dropout_prob=0.5):
        # Call to the parent class constructor (nn.Module)
        super(custom_Resnet, self).__init__()
        # Initialize the number of input channels for the first convolutional layer
        self.in_channels = n_start_filters

        # Define the first layer of the network
        self.layer1 = nn.Sequential(
            # Initial convolutional layer to capture features from input images
            nn.Conv2d(3, n_start_filters, kernel_size=3, padding=1, bias=False),
            # Batch normalization layer to stabilize and speed up training
            nn.BatchNorm2d(n_start_filters),
            # ReLU activation function applied in-place to save memory
            nn.ReLU(inplace=True)
        )

        # Create layer 2 of the network using the custom make_layer method
        self.layer2 = self.make_layer(block, n_start_filters, layers[0], stride=1)
        # Create layer 3 of the network, doubling the number of filters and using a stride of 2 for downsampling
        self.layer3 = self.make_layer(block, n_start_filters * 2, layers[1], stride=2)
        # Create layer 4 of the network, further doubling the filters for more complex features, with stride 2
        self.layer4 = self.make_layer(block, n_start_filters * 4, layers[2], stride=2)

        # Dropout layer to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training
        self.dropout = nn.Dropout(dropout_prob)
        # Adaptive average pooling layer to output a fixed size tensor, useful for different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer to map the learned features to the classes
        self.fc = nn.Linear(self.in_channels, num_classes)

    def make_layer(self, block, out_channels, n_blocks, stride):
        # Define a method to create a layer of blocks for the ResNet
        layers = []
        # First block with the possibility of changing stride and channels
        layers.append(block(self.in_channels, out_channels, stride))
        # Update in_channels for use in the next blocks
        self.in_channels = out_channels
        # Extend the list with additional blocks, keeping the number of channels and using the default stride
        layers.extend([block(out_channels, out_channels) for i in range(1, n_blocks)])
        return nn.Sequential(*layers)

    def forward(self, x):
        # Define the forward pass through the network
        out = self.layer1(x)  # Initial layer
        out = self.layer2(out)  # Second layer
        out = self.layer3(out)  # Third layer
        out = self.layer4(out)  # Fourth layer
        out = self.avgpool(out)  # Global average pooling
        out = out.view(out.size(0), -1)  # Flatten the output for the fully connected layer
        out = self.dropout(out)  # Apply dropout
        out = self.fc(out)  # Fully connected layer to output the class scores
        return out


   model = custom_Resnet(Resnet_block, 32, [13, 13, 13], 10, 0.5).to(device)
   model.load_state_dict(torch.load('best_model3.pth'))
   model = model.to(device)
   
   ```

3. Test on your dataset with the augmentations and normaliation mentioned in the ipynb
