from torch import sigmoid
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet2d(nn.Module):
    def __init__(self):
        super(ResNet2d, self).__init__()

        #   Vanilla ResNet-18 expects input tensors of size (3, 224, 224)
        #   and has Linear layer with 1000 neurons as output layer
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        #   Change first layer to expect 1 channel tensor
        #   first conv layer was: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        #   Replace output layer with 1 output neuron and connect a Sigmoid function to it
        in_features_output_layer = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=in_features_output_layer, out_features=1, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        x = sigmoid(x)
        return x
