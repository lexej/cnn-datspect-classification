import glob
import sys

import random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torchvision.transforms import InterpolationMode

from model import BasicModel

ROOT_PATH = Path('/Users/aleksej/IdeaProjects/master-thesis-kucerenko/UKE_SPECT')
IMG_2D_RIGID_PATH = ROOT_PATH.joinpath('rigid', '2d')
LABELS_PATH = ROOT_PATH.joinpath('scans_20230424.xlsx')

RANDOM_SEED = 1234

# Set all seeds

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)  # if using multiple GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
generator = torch.Generator()
generator.manual_seed(RANDOM_SEED)

sklearn.utils.check_random_state(RANDOM_SEED)


def get_data():

    # -----------------------------------------------------------------------------------------------------------
    #   Features:

    #  Resize images from (91, 109) to size (64, 64) using bicubic interpolation
    # -> Idea: Generalize to future cases where image size may change
    #  Interpolation method has impact on Precision and Recall !!!
    #        -> Bilinear for better Precision; Bicubic for better Recall, NEAREST_EXACT makes trade-off
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(64, 64), interpolation=InterpolationMode.BICUBIC, antialias=True),
        # transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize image
    ])

    file_paths = sorted(glob.glob(f'{str(IMG_2D_RIGID_PATH)}/*.nii'))

    X = []

    for fp in file_paths:
        data_nifti = nib.load(fp)

        img_nifti = data_nifti.get_fdata()

        img_tensor = img_transforms(img_nifti)

        X.append(img_tensor)

    X = torch.stack(X)
    # print(features.size())
    # plt.imshow(features[0][0], cmap='hot')
    # plt.show()

    # -----------------------------------------------------------------------------------------------------------

    #   Labels:

    labels_raw_table = pd.read_excel(LABELS_PATH)

    #   For now: Compute column with majority values among each rater labels
    labels_raw_table['y'] = labels_raw_table[['R1', 'R2', 'R3']].mode(axis=1)[0]

    labels = labels_raw_table[['ID', 'y']].sort_values(by=['ID'])

    y = torch.from_numpy(labels['y'].to_numpy())

    # Add channel dimension (required for loss function)
    y = y.unsqueeze(1)

    # -----------------------------------------------------------------------------------------------------------

    return X.float(), y.float()


def normalize_data_splits(X_train, X_test, X_validation):
    #  Calculate mean and std along all train examples and along all pixels (-> scalar)
    X_train_mean = torch.mean(X_train, dim=(0, 2, 3))
    X_train_std = torch.std(X_train, dim=(0, 2, 3))

    #  Normalization of all splits

    X_train_normalized = (X_train - X_train_mean.reshape(1, 1, 1, 1)) / X_train_std.reshape(1, 1, 1, 1)
    X_test_normalized = (X_test - X_train_mean.reshape(1, 1, 1, 1)) / X_train_std.reshape(1, 1, 1, 1)
    X_validation_normalized = (X_validation - X_train_mean.reshape(1, 1, 1, 1)) / X_train_std.reshape(1, 1, 1, 1)

    return X_train_normalized, X_test_normalized, X_validation_normalized


def train_model(model, num_epochs, train_dataloader, validation_dataloader, optimizer, loss_fn):
    for epoch in range(num_epochs):
        # 1. Epoch training

        model.train()
        train_loss = 0.0

        for batch_features, batch_labels in train_dataloader:
            # Forward pass
            outputs = model(batch_features)
            loss = loss_fn(outputs, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Average loss for epoch
        train_loss /= len(train_dataloader)

        # 2. Epoch validation

        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for val_features, val_labels in validation_dataloader:
                val_outputs = model(val_features)
                val_loss = loss_fn(val_outputs, val_labels)
                validation_loss += val_loss.item()

        # Average loss for epoch
        validation_loss /= len(validation_dataloader)

        # 3. Print epoch results

        print(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {train_loss}, Valid loss: {validation_loss}")

    return model


def evaluate_on_test_data(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)

            # Convert sigmoid outputs to predicted labels
            predicted_labels = torch.round(outputs)

            # Append predicted labels and true labels to lists
            predictions.extend(predicted_labels.tolist())
            true_labels.extend(batch_labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    print('\nEvaluation on test data:\n')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")

    conf_matrix = confusion_matrix(true_labels, predictions)
    classific_report = classification_report(true_labels, predictions, digits=4)

    print('\nConfusion matrix:')
    print(conf_matrix)
    print('\nClassification report:')
    print(classific_report)


def main():
    X, y = get_data()

    #   train-test split (use stratify to ensure equal distribution)
    #   -> stratify has made ENORMOUS improve in convergence
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.4,
                                                        random_state=RANDOM_SEED,
                                                        shuffle=True,
                                                        stratify=y)

    #  test-validation split (use stratify to ensure equal distribution)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test,
                                                                  test_size=0.5,
                                                                  random_state=RANDOM_SEED,
                                                                  shuffle=True,
                                                                  stratify=y_test)

    #  Normalize all splits
    X_train, X_test, X_validation = normalize_data_splits(X_train, X_test, X_validation)

    # --------------------------------------------------------
    #   Model training part:

    batch_size = 32

    train_dataloader = DataLoader(dataset=TensorDataset(X_train, y_train),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  generator=generator)

    validation_dataloader = DataLoader(TensorDataset(X_validation, y_validation),
                                       batch_size=batch_size,
                                       shuffle=False,
                                       generator=generator)

    model = BasicModel()
    model.initialize_weights()

    loss_fn = nn.BCELoss()
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 30

    model = train_model(model,
                        num_epochs,
                        train_dataloader,
                        validation_dataloader,
                        optimizer,
                        loss_fn)

    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=batch_size,
                             shuffle=False,
                             generator=generator)

    evaluate_on_test_data(model, test_loader)


if __name__ == '__main__':
    main()


