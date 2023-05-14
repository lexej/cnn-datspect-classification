import glob
import os

import yaml
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import nibabel as nib
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torchvision.transforms import InterpolationMode

from src.model.model_2d import BaselineModel2d


config_file = 'config.yaml'

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

IMG_2D_RIGID_PATH = Path(config['paths']['images']['rigid']['2d'])
LABELS_PATH = Path(config['paths']['labels_file'])

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


def get_data(target_input_height, target_input_width):

    # -----------------------------------------------------------------------------------------------------------
    #   Features:

    #  Resize images from (91, 109) to size (input_height, input_width) using bicubic interpolation
    # -> Idea: Generalize to future cases where image size may change
    #  Interpolation method has impact on Precision and Recall !!!
    #        -> Bilinear for better Precision; Bicubic for better Recall, NEAREST_EXACT makes trade-off
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(target_input_height, target_input_width),
                          interpolation=InterpolationMode.BICUBIC,
                          antialias=True),
        # transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize image
    ])

    file_paths = sorted(glob.glob(f'{str(IMG_2D_RIGID_PATH)}/*.{config["data"]["file_format"]}'))

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
    best_loss = float('inf')
    best_weights = None
    best_epoch = None

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

        # Save weights if best
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_weights = model.state_dict()
            best_epoch = epoch+1

        # 3. Print epoch results

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train loss: {round(train_loss, 4)}, "
              f"Valid loss: {round(validation_loss, 4)}")

    results_path = 'results/best_weights.pth'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    torch.save(best_weights, results_path)

    return model, best_epoch


def evaluate_on_test_data(model, best_epoch, test_loader):

    # Load weights
    model.load_state_dict(torch.load('results/best_weights.pth'))

    #   Evaluation mode
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)

            # Probabilities -> predicted labels (for now: 0.5 decision boundary)
            predicted_labels = torch.round(outputs)

            preds.extend(predicted_labels.tolist())
            trues.extend(batch_labels.tolist())

    acc_score = accuracy_score(trues, preds)
    precision = precision_score(trues, preds)
    recall = recall_score(trues, preds)
    f1 = f1_score(trues, preds)
    conf_matrix = confusion_matrix(trues, preds)
    class_report = classification_report(trues, preds, digits=4)

    print(f'\nEvaluation of model (best epoch: {best_epoch}) on test split:\n')
    print(f"Accuracy: {acc_score}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print('\nConfusion matrix:')
    print(conf_matrix)
    print('\nClassification report:')
    print(class_report)


def main():

    (input_height, input_width) = config['data']['preprocessing']['target_img_size']

    X, y = get_data(target_input_height=input_height, target_input_width=input_width)

    #   train-test split (use stratify to ensure equal distribution)
    #   -> stratify has made ENORMOUS improve in convergence
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=config['data']['test_to_train_split_size_percent'],
                                                        random_state=RANDOM_SEED,
                                                        shuffle=True,
                                                        stratify=y)

    #  test-validation split (use stratify to ensure equal distribution)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test,
                                                                  test_size=config['data']['valid_to_test_split_size_percent'],
                                                                  random_state=RANDOM_SEED,
                                                                  shuffle=True,
                                                                  stratify=y_test)

    #  Normalize all splits
    X_train, X_test, X_validation = normalize_data_splits(X_train, X_test, X_validation)

    # --------------------------------------------------------
    #   Model training part:

    batch_size = config['model']['training_params']['batch_size']

    train_dataloader = DataLoader(dataset=TensorDataset(X_train, y_train),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  generator=generator)

    validation_dataloader = DataLoader(TensorDataset(X_validation, y_validation),
                                       batch_size=batch_size,
                                       shuffle=False,
                                       generator=generator)

    model = BaselineModel2d(input_height=input_height, input_width=input_height)
    model.initialize_weights()

    loss_fn = nn.BCELoss()
    lr = config['model']['training_params']['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = config['model']['training_params']['epochs']

    model, best_epoch = train_model(model,
                                    num_epochs,
                                    train_dataloader,
                                    validation_dataloader,
                                    optimizer,
                                    loss_fn)

    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=batch_size,
                             shuffle=False,
                             generator=generator)

    evaluate_on_test_data(model, best_epoch, test_loader)


if __name__ == '__main__':
    main()


