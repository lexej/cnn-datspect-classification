import glob
import sys

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import BasicModel

ROOT_PATH = Path('/Users/aleksej/IdeaProjects/master-thesis-kucerenko/UKE_SPECT')
IMG_2D_RIGID_PATH = ROOT_PATH.joinpath('rigid', '2d')
LABELS_PATH = ROOT_PATH.joinpath('scans_20230424.xlsx')

RANDOM_SEED = 1234


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
    #
    labels = labels.view(-1, 1)

    return features, labels


def main():
    features, labels = get_data()
    features = features.float()
    labels = labels.float()

    #   train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.3,
                                                        random_state=RANDOM_SEED,
                                                        shuffle=True,)

    # test-validation split
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test,
                                                                  test_size=0.5,
                                                                  random_state=RANDOM_SEED,
                                                                  shuffle=True,)

    dataset = TensorDataset(X_train, y_train)

    batch_size = 32

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(dataloader)

    model = BasicModel()
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_features, batch_labels in dataloader:
            # Set model to train mode
            model.train()

            # Forward pass
            outputs = model(batch_features)
            loss = loss_fn(outputs, batch_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss for printing
            running_loss += loss.item()

        # Average loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")


if __name__ == '__main__':
    main()


