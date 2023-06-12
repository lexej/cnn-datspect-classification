from torch import sigmoid, softmax
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet2d(nn.Module):
    def __init__(self, num_out_features: int, outputs_function: str):
        super(ResNet2d, self).__init__()

        self.outputs_function = outputs_function

        #   Vanilla ResNet-18 expects input tensors of size (3, 224, 224)
        #   and has Linear layer with 1000 neurons as output layer
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        #   Change first layer to expect 1 channel tensor
        #   first conv layer was: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        #   Replace output layer with 1 output neuron and connect a Sigmoid function to it
        in_features_output_layer = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=in_features_output_layer, out_features=num_out_features, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        if self.outputs_function == 'sigmoid':
            x = sigmoid(x)
        elif self.outputs_function == 'softmax':
            x = softmax(x, dim=1)
        return x
