import os
import json
import shutil
import yaml
import argparse
import sys
import re
import shutil

from typing import List

from tqdm import tqdm

import numpy as np
import pandas as pd

import nibabel as nib

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import sigmoid, softmax
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve, auc

#   Seeding

RANDOM_SEED = 42  # 72328191

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

#   Set device based on availability
if torch.cuda.is_available():
    device = torch.device("cuda")

    # Set seeds
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif torch.backends.mps.is_available():
    device = torch.device("mps")

    #   Set seeds
    torch.mps.manual_seed(RANDOM_SEED)
    torch.backends.mps.deterministic = True
    torch.backends.mps.benchmark = False
else:
    raise Exception("something went wrong")


generator = torch.Generator()
generator.manual_seed(RANDOM_SEED)

sklearn.utils.check_random_state(RANDOM_SEED)


