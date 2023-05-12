import glob
import torch
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt


ROOT_PATH = Path('/Users/aleksej/IdeaProjects/master-thesis-kucerenko/UKE_SPECT')
IMG_2D_RIGID_PATH = ROOT_PATH.joinpath('rigid', '2d')
LABELS_PATH = ROOT_PATH.joinpath('scans_20230424.xlsx')


def get_data():
    transform = transforms.ToTensor()

    file_paths = sorted(glob.glob(f'{str(IMG_2D_RIGID_PATH)}/*.nii'))

    features = []

    for fp in file_paths:
        data_nifti = nib.load(fp)

        img_nifti = data_nifti.get_fdata()

        img_tensor = transform(img_nifti)

        features.append(img_tensor)

    features = torch.stack(features)
    # print(features.size())
    # plt.imshow(features[0][0], cmap='hot')
    # plt.show()

    labels_df = pd.read_excel(LABELS_PATH)
    labels = labels_df[['ID', 'R1', 'R2', 'R3']].sort_values(by=['ID'])
    # use only rater 'R1' for now
    labels = labels['R1'].to_numpy()
    labels = torch.from_numpy(labels)

    return features, labels


def main():
    features, labels = get_data()
    print(features.size())
    print(labels.size())


if __name__ == '__main__':
    main()


