import os
import glob
import json
import sys

import yaml
import pandas as pd

import matplotlib.pyplot as plt

import nibabel as nib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay

from src.model.model_2d import BaselineModel2d


RANDOM_SEED = 1327

# Set all seeds

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
generator = torch.Generator()
generator.manual_seed(RANDOM_SEED)

sklearn.utils.check_random_state(RANDOM_SEED)


def run_experiment(experiment_name: str, config: dict):

    images_dirpath = config['paths']['images']
    labels_filepath = config['paths']['labels_file']

    def get_data_splits(target_input_height, target_input_width):
        # -----------------------------------------------------------------------------------------------------------
        #   Get the features

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

        file_paths = sorted(glob.glob(f'{images_dirpath}/*.{config["data"]["file_format"]}'))

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

        #   Get the labels

        labels_raw_table = pd.read_excel(labels_filepath)

        #   For now: Compute column with majority values among each rater labels
        labels_raw_table['y'] = labels_raw_table[['R1', 'R2', 'R3']].mode(axis=1)[0]

        labels = labels_raw_table[['ID', 'y']].sort_values(by=['ID'])

        y = torch.from_numpy(labels['y'].to_numpy())

        # Add channel dimension (required for loss function)
        y = y.unsqueeze(1)

        #   Convert Torch tensors to float type

        X = X.float()
        y = y.float()

        # -----------------------------------------------------------------------------------------------------------

        #   Splitting the data

        #   1. train-test split (stratify to ensure equal distribution)
        #   -> stratify has made ENORMOUS improve in convergence
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=config['data']['test_to_train_split_size_percent'],
                                                            random_state=RANDOM_SEED,
                                                            shuffle=True,
                                                            stratify=y)

        #  2. test-validation split (stratify to ensure equal distribution)
        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test,
                                                                      test_size=config['data'][
                                                                          'valid_to_test_split_size_percent'],
                                                                      random_state=RANDOM_SEED,
                                                                      shuffle=True,
                                                                      stratify=y_test)

        #  Normalize all splits

        X_train, X_test, X_validation = normalize_data_splits(X_train, X_test, X_validation)

        return X_train, y_train, X_validation, y_validation, X_test, y_test

    def normalize_data_splits(X_train, X_test, X_validation):
        #  Calculate mean and std along all train examples and along all pixels (-> scalar)
        X_train_mean = torch.mean(X_train, dim=(0, 2, 3))
        X_train_std = torch.std(X_train, dim=(0, 2, 3))

        #  Normalization of all splits

        X_train_normalized = (X_train - X_train_mean.reshape(1, 1, 1, 1)) / X_train_std.reshape(1, 1, 1, 1)
        X_test_normalized = (X_test - X_train_mean.reshape(1, 1, 1, 1)) / X_train_std.reshape(1, 1, 1, 1)
        X_validation_normalized = (X_validation - X_train_mean.reshape(1, 1, 1, 1)) / X_train_std.reshape(1, 1, 1, 1)

        return X_train_normalized, X_test_normalized, X_validation_normalized

    def train_model(model, num_epochs, train_dataloader, valid_dataloader, optimizer, loss_fn):
        best_loss = float('inf')
        best_weights = None
        best_epoch = None

        train_losses = []
        valid_losses = []

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

            # Average epoch loss
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)

            # 2. Epoch validation

            model.eval()
            validation_loss = 0.0

            with torch.no_grad():
                for val_features, val_labels in valid_dataloader:
                    val_outputs = model(val_features)
                    val_loss = loss_fn(val_outputs, val_labels)
                    validation_loss += val_loss.item()

            # Average epoch loss
            validation_loss /= len(valid_dataloader)
            valid_losses.append(validation_loss)

            # Save weights if best
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_weights = model.state_dict()
                best_epoch = epoch + 1

            # 3. Print epoch results

            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train loss: {round(train_loss, 4)}, "
                  f"Valid loss: {round(validation_loss, 4)}")

        best_weights_path = f'results/{experiment_name}/training/best_weights.pth'
        os.makedirs(os.path.dirname(best_weights_path), exist_ok=True)
        torch.save(best_weights, best_weights_path)

        #   plot loss curves
        plt.figure()
        plt.plot(range(1, num_epochs + 1), train_losses, label='train loss')
        plt.plot(range(1, num_epochs + 1), valid_losses, label='valid loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'results/{experiment_name}/training/training_curves.png', dpi=300)

        return model, best_epoch, best_weights_path

    def evaluate_on_test_data(model, best_epoch_weights_path: str, best_epoch: int, test_dataloader):

        # Load weights
        model.load_state_dict(torch.load(best_epoch_weights_path))

        #   Evaluation mode
        model.eval()

        preds = []
        trues = []

        with torch.no_grad():
            for batch_features, batch_labels in test_dataloader:
                outputs = model(batch_features)

                # Probabilities -> predicted labels (for now: 0.5 decision boundary)
                predicted_labels = torch.round(outputs)

                preds.extend(predicted_labels.tolist())
                trues.extend(batch_labels.tolist())

        acc_score = accuracy_score(trues, preds)
        precision = precision_score(trues, preds)
        recall = recall_score(trues, preds)
        f1 = f1_score(trues, preds)
        conf_matrix = confusion_matrix(trues, preds, labels=[0, 1])

        print(f'\nEvaluation of model (best epoch: {best_epoch}) on test split:\n')
        print(f"Accuracy: {acc_score}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        print('\nConfusion matrix:')
        print(conf_matrix)

        # Save results

        results_test_split = {
            'accuracy': acc_score,
            'precision': precision,
            'recall': recall,
            'f1-score': f1
        }

        subdir = f'results/{experiment_name}/testing'
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        with open(os.path.join(subdir, 'results_test_split.json'), 'w') as file:
            json.dump(results_test_split, file, indent=4)

        cm_display = ConfusionMatrixDisplay(conf_matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.savefig(f'results/{experiment_name}/testing/conf_matrix_test_split.png')

    # ---------------------------------------------------------------------------------------------------------

    print(f'\nExperiment "{experiment_name}": \n')

    #   Run experiment:

    (input_height, input_width) = config['data']['preprocessing']['target_img_size']
    batch_size = config['model']['training_params']['batch_size']
    lr = config['model']['training_params']['lr']
    num_epochs = config['model']['training_params']['epochs']

    #   Create data splits

    data_splits = get_data_splits(target_input_height=input_height,
                                  target_input_width=input_width)

    X_train, y_train, X_validation, y_validation, X_test, y_test = data_splits

    #   Create data loaders

    train_dataloader = DataLoader(dataset=TensorDataset(X_train, y_train),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  generator=generator)

    valid_dataloader = DataLoader(TensorDataset(X_validation, y_validation),
                                  batch_size=batch_size,
                                  shuffle=False,
                                  generator=generator)

    test_dataloader = DataLoader(TensorDataset(X_test, y_test),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 generator=generator)

    #   Initialize and train model

    model = BaselineModel2d(input_height=input_height, input_width=input_height)
    model.initialize_weights()

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    model, best_epoch, best_epoch_weights_path = train_model(model,
                                                             num_epochs,
                                                             train_dataloader,
                                                             valid_dataloader,
                                                             optimizer,
                                                             loss_fn)

    #   Evaluate model on test data

    evaluate_on_test_data(model, best_epoch_weights_path, best_epoch, test_dataloader)

    print('-----------------------------------------------------------------------------------------')


def main():

    #   Path relative to current file
    configs_dirpath = 'configs'

    #   Run experiment for rigid case and 2D images:

    config_rigid_2d_path = f'{configs_dirpath}/config_rigid_2d.yaml'

    with open(config_rigid_2d_path, 'r') as f:
        config_rigid_2d = yaml.safe_load(f)

    run_experiment(experiment_name='rigid_2d',
                   config=config_rigid_2d)

    #   Run experiment for affine case and 2D images:

    config_affine_2d_path = f'{configs_dirpath}/config_affine_2d.yaml'

    with open(config_affine_2d_path, 'r') as f:
        config_affine_2d = yaml.safe_load(f)

    run_experiment(experiment_name='affine_2d',
                   config=config_affine_2d)


if __name__ == '__main__':
    main()
