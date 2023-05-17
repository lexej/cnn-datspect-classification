import torch.nn as nn


class BaselineModel2d(nn.Module):
    def __init__(self, input_height: int, input_width: int):
        super(BaselineModel2d, self).__init__()

        #   Convolution layers
        #       - Stride and padding of Conv layers chosen so that spatial dimensions are preserved

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        #   Activation function

        self.relu = nn.ReLU()

        #   Pooling for dimensionality reduction

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # BatchNorm ?? TODO

        self.flatten = nn.Flatten()

        #   Dense layers

        # Dimensionality reduction only through the application of 3 MaxPool2d layers (each divides size by 2)
        self.linear1 = nn.Linear(128 * (input_height // (2*2*2)) * (input_width // (2*2*2)), 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        x = self.sigmoid(x)

        return x

    def initialize_weights(self):
        #   initialize the weights of each Convolution and Linear layer
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Use He initialization
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
