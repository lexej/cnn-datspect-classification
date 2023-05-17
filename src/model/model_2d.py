import torch.nn as nn


class BaselineModel2d(nn.Module):
    def __init__(self, input_height: int, input_width: int):
        super(BaselineModel2d, self).__init__()

        #   Convolution layers
        #       - Stride and padding of Conv layers chosen so that spatial dimensions are preserved

        #   ReLU used as activation function

        #   Pooling layers for dimensionality reduction (each layer divides size by 2)

        self.layers = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #   Flatten Conv layer
            nn.Flatten(),
            #   Dense layers
            nn.Linear(128 * (input_height // (2*2*2)) * (input_width // (2*2*2)), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Dropout(0.1)   # -> leads to worse performance on test set in rigid case
            nn.Sigmoid()
        ])

        # BatchNorm ?? TODO

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def initialize_weights(self):
        #   initialize the weights of each Convolution and Linear layer
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Use He initialization
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
