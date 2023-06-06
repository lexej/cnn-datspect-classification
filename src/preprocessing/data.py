import os

import nibabel as nib

import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset


RANDOM_SEED = 1327

torch.manual_seed(RANDOM_SEED)


class SpectDataset(Dataset):
    def __init__(self, features_dirpath, labels_filepath, target_input_height,
                 target_input_width, interpolation_method):
        self.features_dirpath = features_dirpath
        self.labels_filepath = labels_filepath

        self.target_input_height = target_input_height
        self.target_input_width = target_input_width
        self.interpolation_method = interpolation_method

        self.labels = pd.read_excel(labels_filepath)

        #   Sort filenames by name
        self.image_filenames = sorted(os.listdir(features_dirpath))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        #   ---------------------------------------------------------------------------
        #   Get image

        image_filename = self.image_filenames[idx]

        image_path = os.path.join(self.features_dirpath, image_filename)

        data_nifti = nib.load(image_path)

        img = data_nifti.get_fdata()

        img = self.apply_transforms(img)

        #   ---------------------------------------------------------------------------
        #   Get label

        label_row = self.labels.loc[self.labels['ID'].astype(int) == idx+1]

        #   Convert pandas DataFrame row to pandas Series of labels of interest
        label_row = label_row[['R1', 'R2', 'R3']].reset_index(drop=True).squeeze()

        #   Majority value of the pandas Series

        label = int(label_row.mode()[0])

        #   To torch tensor with additional dimension
        label = torch.tensor(label).unsqueeze(0)

        label = label.float()

        return img, label

    def apply_transforms(self, img: np.ndarray) -> torch.Tensor:

        #   1. Crop image of size (91, 109) to region of interest (91, 91)
        #       policy: select all rows and 10th to 100th column

        transformed_img = img[:, 9:100]

        #   2. Convert to torch tensor
        #   3. Resize images from (91, 91) to target size (input_height, input_width)
        #           using certain interpolation method
        #       - Attention: Interpolation method has impact on model performance !

        img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(self.target_input_height, self.target_input_width),
                              interpolation=getattr(InterpolationMode, self.interpolation_method.upper()),
                              antialias=True)
        ])

        transformed_img = img_transforms(transformed_img)

        #   4. Convert to float type

        transformed_img = transformed_img.float()

        return transformed_img
