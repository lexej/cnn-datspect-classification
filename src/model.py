from common import nn
from common import resnet18, ResNet18_Weights
from common import resnet34, ResNet34_Weights


class ResNet18(nn.Module):
    def __init__(self, num_out_features: int, outputs_activation_func, pretrained: bool):
        super(ResNet18, self).__init__()

        self.outputs_activation_func = outputs_activation_func

        #   Vanilla ResNet-18 expects input tensors of size (3, 224, 224)
        #   and has Linear layer with 1000 neurons as output layer
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.resnet = resnet18(weights=weights)

        #   Change first layer to expect 1 channel tensor
        #   first conv layer was: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        #   Replace output layer with 1 output neuron and connect a Sigmoid function to it
        in_features_output_layer = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=in_features_output_layer, out_features=num_out_features, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        if self.outputs_activation_func is not None:
            x = self.outputs_activation_func(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, num_out_features: int, outputs_activation_func, pretrained: bool):
        super(ResNet34, self).__init__()

        self.outputs_activation_func = outputs_activation_func

        #   Vanilla ResNet-34 expects input tensors of size (3, 224, 224)
        #   and has Linear layer with 1000 neurons as output layer
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.resnet = resnet34(weights=weights)

        #   Change first layer to expect 1 channel tensor
        #   first conv layer was: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        #   Replace output layer with 1 output neuron and connect a Sigmoid function to it
        in_features_output_layer = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=in_features_output_layer, out_features=num_out_features, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        if self.outputs_activation_func is not None:
            x = self.outputs_activation_func(x)
        return x


class CustomModel2d(nn.Module):
    def __init__(self, input_height: int, input_width: int):
        super(CustomModel2d, self).__init__()

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
