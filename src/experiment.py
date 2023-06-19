import os
import json
import shutil
import yaml
import argparse
import sys

from typing import List
from tqdm import tqdm

import numpy as np
import pandas as pd

import nibabel as nib

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve, auc

from model.custom_model_2d import CustomModel2d
from model.resnet18_2d import ResNet18
from model.resnet34_2d import ResNet34


#   Parse arguments

parser = argparse.ArgumentParser(description='Script for running experiments using a yaml config.')

parser.add_argument('-c', '--config', type=str, required=True, help='Path to the YAML config file')

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

config_filename = os.path.basename(args.config).removesuffix('.yaml')

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


class Encoder:

    def __init__(self, strategy: int):
        self.strategy = strategy

    def get_label_using_strategy(self, final_labels: pd.Series, session_labels: pd.Series):

        final_labels_occurrences = final_labels.value_counts()
        session_labels_occurrences = session_labels.value_counts()

        count_of_0_in_final_labels = final_labels_occurrences.get(0, default=0)
        count_of_0_in_session_labels = session_labels_occurrences.get(0, default=0)
        count_of_1_in_session_labels = session_labels_occurrences.get(1, default=0)

        if self.strategy == 0:
            #   Binary classification case
            #   Pick random label from the 3 final labels
            label = np.random.choice(list(final_labels.values))
        elif self.strategy == 1:
            #   3 classes: "normal", "uncertain", "reduced"

            #   Only consider final labels

            if count_of_0_in_final_labels == 3:
                #   all raters agree on label "0"
                label = 0
            elif count_of_0_in_final_labels == 2 or count_of_0_in_final_labels == 1:
                #   rater do not agree on label
                label = 1
            elif count_of_0_in_final_labels == 0:
                #   all raters agree on label "1"
                label = 2
        elif self.strategy == 2:
            #   7 classes; ordinal classification problem assumed

            #   0: "most likely normal"               <-> [1, 0, 0, 0, 0, 0, 0]
            #   1: "probably normal"                  <-> [1, 1, 0, 0, 0, 0, 0]
            #   2: "more likely normal than reduced"  <-> [1, 1, 1, 0, 0, 0, 0]
            #   3: "uncertain"                        <-> [1, 1, 1, 1, 0, 0, 0]
            #   4: "more likely reduced than normal"  <-> [1, 1, 1, 1, 1, 0, 0]
            #   5: "probably reduced"                 <-> [1, 1, 1, 1, 1, 1, 0]
            #   6: "most likely reduced"              <-> [1, 1, 1, 1, 1, 1, 1]

            #   Consider both final and session labels

            if count_of_0_in_final_labels == 3:
                if count_of_0_in_session_labels > 3:
                    #   most likely normal
                    label = 0
                else:
                    #   probably normal
                    label = 1
            elif count_of_0_in_final_labels == 2:
                if count_of_0_in_session_labels > 3:
                    #   more likely normal than reduced
                    label = 2
                else:
                    #   uncertain
                    label = 3
            elif count_of_0_in_final_labels == 1:
                if count_of_1_in_session_labels > 3:
                    #   more likely reduced than normal
                    label = 4
                else:
                    #   uncertain
                    label = 3
            else:
                #   Case where: count_of_0_in_final_labels = 0 <=> count_of_1_in_final_labels = 3
                if count_of_1_in_session_labels > 3:
                    label = 6
                else:
                    label = 5

        label = self.__encode_integer_label(label)

        return label

    def __encode_integer_label(self, label: int) -> torch.Tensor:

        if self.strategy == 0:
            #   vanilla binary classification; nn.BCELoss() expects integer ground truths
            encoded_label = [label]
            dtype = torch.float32
        elif self.strategy == 1:
            #   vanilla multi-class classification; nn.CrossEntropyLoss() expects ground truths with type integer
            encoded_label = label
            dtype = torch.int32
        elif self.strategy == 2:
            dtype = torch.float32
            if label == 0:
                encoded_label = [1, 0, 0, 0, 0, 0, 0]
            elif label == 1:
                encoded_label = [1, 1, 0, 0, 0, 0, 0]
            elif label == 2:
                encoded_label = [1, 1, 1, 0, 0, 0, 0]
            elif label == 3:
                encoded_label = [1, 1, 1, 1, 0, 0, 0]
            elif label == 4:
                encoded_label = [1, 1, 1, 1, 1, 0, 0]
            elif label == 5:
                encoded_label = [1, 1, 1, 1, 1, 1, 0]
            elif label == 6:
                encoded_label = [1, 1, 1, 1, 1, 1, 1]
            else:
                encoded_label = None

        encoded_label = torch.tensor(encoded_label, dtype=dtype, device=device)

        return encoded_label

    @staticmethod
    def decode_labels(labels: torch.Tensor):
        #   Function used only for strategy = 2

        #   labels should have shape (batch_size, num_classes)

        labels_decoded = torch.zeros((labels.shape[0], 1))

        for i, val in enumerate(labels):
            encoded_val = list(val)
            if encoded_val == [1, 0, 0, 0, 0, 0, 0]:
                labels_decoded[i][0] = 0
            elif encoded_val == [1, 1, 0, 0, 0, 0, 0]:
                labels_decoded[i][0] = 1
            elif encoded_val == [1, 1, 1, 0, 0, 0, 0]:
                labels_decoded[i][0] = 2
            elif encoded_val == [1, 1, 1, 1, 0, 0, 0]:
                labels_decoded[i][0] = 3
            elif encoded_val == [1, 1, 1, 1, 1, 0, 0]:
                labels_decoded[i][0] = 4
            elif encoded_val == [1, 1, 1, 1, 1, 1, 0]:
                labels_decoded[i][0] = 5
            elif encoded_val == [1, 1, 1, 1, 1, 1, 1]:
                labels_decoded[i][0] = 6

        return labels_decoded

    @staticmethod
    def decode_preds(preds: torch.Tensor):
        #   Function used only for strategy = 2

        preds_decoded = torch.zeros((preds.shape[0], 1))

        #   threshold for each predicted output neuron
        threshold = 0.5

        for i, pred in enumerate(preds):
            for j, val in enumerate(pred):
                if j == 0:
                    if val > threshold:
                        #   Do nothing since preds_decoded is initialized with zeros
                        continue
                    else:
                        raise Exception("something went wrong")

                if val > threshold:
                    preds_decoded[i][0] += 1
                else:
                    break

        return preds_decoded


class SpectDataset(Dataset):
    def __init__(self, features_dirpath, labels_filepath, target_input_height,
                 target_input_width, interpolation_method, enc: Encoder):
        self.features_dirpath = features_dirpath
        self.labels_filepath = labels_filepath

        self.target_input_height = target_input_height
        self.target_input_width = target_input_width
        self.interpolation_method = interpolation_method

        self.enc = enc

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

        #   TODO: Frage an Ralph: Sind die Bezeichnungen der .nii files die IDs wie in der Excel ???
        #    Ich habs angenommen!!

        id_ = idx + 1

        label_row = self.labels.loc[self.labels['ID'].astype(int) == id_]

        #   Convert pandas DataFrame row to pandas Series of labels of interest

        labels_full = label_row[['R1', 'R2', 'R3',
                                 'R1S1', 'R1S2', 'R2S1',
                                 'R2S2', 'R3S1', 'R3S2']].reset_index(drop=True).squeeze()

        final_labels = labels_full[['R1', 'R2', 'R3']]

        session_labels = labels_full[['R1S1', 'R1S2', 'R2S1', 'R2S2', 'R3S1', 'R3S2']]

        label = self.enc.get_label_using_strategy(final_labels, session_labels)

        metadata = {
            'id': id_,
            'labels_original': labels_full.to_dict()
        }

        return img, label, metadata

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

        #   4. Convert to float type and move to device

        transformed_img = transformed_img.to(device=device, dtype=torch.float)

        return transformed_img


def run_experiment(config: dict):

    # ---------------------------------------------------------------------------------------------------------

    #   Extract parameters from yaml config

    #   Create experiment results directory

    results_dir = os.path.join(os.getcwd(), 'results')

    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, config_filename)

    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.makedirs(results_path)

    images_dirpath = config['images_dir']
    labels_filepath = config['labels_filepath']

    (input_height, input_width) = config['target_img_size']
    interpolation_method = config['interpolation_method']

    model_name = config['model_name']
    pretrained = config['pretrained']
    strategy = config['strategy']

    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['epochs']

    test_to_train_split_size_percent = config['test_to_train_split_size_percent']

    valid_to_train_split_size_percent = config['valid_to_train_split_size_percent']

    # ---------------------------------------------------------------------------------------------------------

    #   Log the config file used for the experiment to results

    with open(os.path.join(results_path, 'config_used.yaml'), 'w') as f:
        yaml.dump(config, f)

    # ---------------------------------------------------------------------------------------------------------

    #   Functions

    def get_dataloaders(target_input_height, target_input_width, interpolation_method: str, enc: Encoder):
        # -----------------------------------------------------------------------------------------------------------
        #   Create dataset

        spect_dataset = SpectDataset(features_dirpath=images_dirpath,
                                     labels_filepath=labels_filepath,
                                     target_input_height=target_input_height,
                                     target_input_width=target_input_width,
                                     interpolation_method=interpolation_method,
                                     enc=enc)

        # -----------------------------------------------------------------------------------------------------------

        #   Splitting the data

        #   1. train-test split  (no stratify used)

        train_indices, test_indices = train_test_split(range(len(spect_dataset)),
                                                       test_size=test_to_train_split_size_percent,
                                                       random_state=RANDOM_SEED,
                                                       shuffle=True)

        train_subset = Subset(dataset=spect_dataset, indices=train_indices)
        test_subset = Subset(dataset=spect_dataset, indices=test_indices)

        #  2. test-validation split  (no stratify used)

        train_indices, val_indices = train_test_split(range(len(train_subset)),
                                                      test_size=valid_to_train_split_size_percent,
                                                      random_state=RANDOM_SEED,
                                                      shuffle=True)

        #   First create validation subset, then reassign train subset !

        val_subset = Subset(dataset=train_subset, indices=val_indices)

        train_subset = Subset(dataset=train_subset, indices=train_indices)

        # -----------------------------------------------------------------------------------------------------------

        #   TODO : Anpassen normalize function ; auch: normalize ja/nein?

        #  Normalize all splits

        #   X_train, X_test, X_validation = normalize_data_splits(X_train, X_test, X_validation)

        # -----------------------------------------------------------------------------------------------------------

        #   Create data loaders for the data subsets

        train_dataloader = DataLoader(dataset=train_subset,
                                      batch_size=batch_size,
                                      shuffle=True,  # True in order to shuffle train data before each new epoch
                                      generator=generator)

        valid_dataloader = DataLoader(dataset=val_subset,
                                      batch_size=batch_size,
                                      shuffle=False)  # Already randomly shuffled

        test_dataloader = DataLoader(dataset=test_subset,
                                     batch_size=batch_size,
                                     shuffle=False)  # Already randomly shuffled

        return train_dataloader, valid_dataloader, test_dataloader

    """
    def normalize_data_splits(X_train, X_test, X_validation):
        #  Calculate mean and std along all train examples and along all pixels (-> scalar)
        X_train_mean = torch.mean(X_train, dim=(0, 2, 3))
        X_train_std = torch.std(X_train, dim=(0, 2, 3))

        #  Normalization of all splits

        X_train_normalized = (X_train - X_train_mean.reshape(1, 1, 1, 1)) / X_train_std.reshape(1, 1, 1, 1)
        X_test_normalized = (X_test - X_train_mean.reshape(1, 1, 1, 1)) / X_train_std.reshape(1, 1, 1, 1)
        X_validation_normalized = (X_validation - X_train_mean.reshape(1, 1, 1, 1)) / X_train_std.reshape(1, 1, 1, 1)

        return X_train_normalized, X_test_normalized, X_validation_normalized
    """

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

            #   tqdm() activates progress bar
            for batch_features, batch_labels, _ in tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}] "):

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
                for val_features, val_labels, _ in valid_dataloader:
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

            print(f" Results for epoch {epoch + 1} - "
                  f"train loss: {round(train_loss, 5)}, "
                  f"valid loss: {round(validation_loss, 5)}")

        results_training_path = os.path.join(results_path, 'training')
        os.makedirs(results_training_path, exist_ok=True)

        best_weights_path = os.path.join(results_training_path, 'best_weights.pth')
        torch.save(best_weights, best_weights_path)

        #   plot loss curves
        plt.figure()
        plt.plot(range(1, num_epochs + 1), train_losses, label='train loss')
        plt.plot(range(1, num_epochs + 1), valid_losses, label='valid loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(results_training_path, 'training_curves.png'), dpi=300)

        return model, best_epoch, best_weights_path

    def evaluate_on_test_data(model, best_epoch_weights_path: str, best_epoch: int, test_dataloader):

        results_testing_path = os.path.join(results_path, 'testing')
        os.makedirs(results_testing_path, exist_ok=True)

        relevant_digits = 5

        # Load weights into model
        model.load_state_dict(torch.load(best_epoch_weights_path))

        #   Evaluation mode
        model.eval()

        preds = []
        trues = []
        ids = []
        labels_original = []

        #   Create numpy array for predictions and ground truths

        with torch.no_grad():
            for batch in test_dataloader:
                batch_features, batch_labels, batch_metadata = batch

                outputs = model(batch_features)

                preds.extend(outputs.tolist())
                trues.extend(batch_labels.tolist())
                ids.extend(batch_metadata['id'].tolist())

                #   Here only use R1, R2 and R3 (for now)
                labels_r_1 = batch_metadata['labels_original']['R1']
                labels_r_2 = batch_metadata['labels_original']['R2']
                labels_r_3 = batch_metadata['labels_original']['R3']

                labels_original.extend(list(zip(labels_r_1, labels_r_2, labels_r_3)))

        preds = np.array(preds)
        trues = np.array(trues)
        ids = np.array(ids)
        labels_original = np.array(labels_original)

        #   TODO: choose threshold_value appropriately
        threshold_value = 0.5

        #   Shape of preds and trues: (num_test_examples, num_classes)

        #   For binary classification: Compute ROC curve and AUC

        if strategy == 0:
            fpr, tpr, thresholds = roc_curve(trues, preds)

            roc_auc = auc(fpr, tpr)

            negative_preds = preds[trues == 0]
            negative_trues = trues[trues == 0]
            positive_preds = preds[trues == 1]
            positive_trues = trues[trues == 1]

            #   ---------------------------------------------------------------------------------------------
            #   Visualize ROC curve and save

            plt.figure(figsize=(12, 6))
            plt.plot(fpr, tpr, label=f'ROC curve; AUC = {round(roc_auc, relevant_digits)}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Evaluation on test data')
            plt.legend(loc='lower right')

            plt.savefig(os.path.join(results_testing_path, 'roc_curve.png'), dpi=300)

            #   ---------------------------------------------------------------------------------------------
            #   Scatter Plot with points representing the test samples (color: label) and x-axis is predicted prob

            plt.figure(figsize=(12, 6))
            plt.scatter(x=negative_preds, y=negative_trues, c='red', label='Negative Ground Truth')
            plt.scatter(x=positive_preds, y=positive_trues, c='blue', label='Positive Ground Truth')
            # Vertical threshold lines
            for x in np.arange(0, 1.1, 0.1):
                plt.axvline(x=x, linestyle='dotted', color='grey')
            plt.xlabel('Predicted Probabilities')
            plt.ylabel('True Labels')
            plt.legend()
            plt.title('Scatter Plot - Evaluation on test data')

            plt.savefig(os.path.join(results_testing_path, 'scatter_plot.png'), dpi=300)

            #   ---------------------------------------------------------------------------------------------
            #   Histogram

            plt.figure(figsize=(12, 6))
            num_bins = 50
            plt.hist(negative_preds, bins=num_bins, alpha=0.7, color='red', label='Negative Ground Truth')
            plt.hist(positive_preds, bins=num_bins, alpha=0.7, color='blue', label='Positive Ground Truth')
            # Vertical threshold lines
            for x in np.arange(0, 1.1, 0.1):
                plt.axvline(x=x, linestyle='dotted', color='grey')
            plt.xlabel('Predicted Probabilities')
            plt.ylabel('Frequency')
            plt.title('Histogram - Evaluation on test data')
            plt.legend()

            plt.savefig(os.path.join(results_testing_path, 'histogram.png'), dpi=300)

            #   ---------------------------------------------------------------------------------------------

            #   Get information about ID's and original labels of misclassified cases (given thresholds)

            #   fp cases

            false_positive_indices = np.where((preds > threshold_value) & (trues == 0))[0]

            fp_samples_dict = {
                'id': ids[false_positive_indices].tolist(),
                'prediction': preds[false_positive_indices].tolist(),
                'label_chosen': trues[false_positive_indices].tolist(),
                'labels_original_r1_r2_r3': labels_original[false_positive_indices].tolist(),
                'threshold_used': threshold_value
            }

            fp_samples = pd.DataFrame(fp_samples_dict)
            fp_samples.to_csv(os.path.join(results_testing_path, 'fp_samples.csv'), index=False)

            #   fn cases

            false_negative_indices = np.where((preds < threshold_value) & (trues == 1))[0]

            fn_samples_dict = {
                'id': ids[false_negative_indices].tolist(),
                'prediction': preds[false_negative_indices].tolist(),
                'label_chosen': trues[false_negative_indices].tolist(),
                'labels_original_r1_r2_r3': labels_original[false_negative_indices].tolist(),
                'threshold_used': threshold_value
            }

            fn_samples = pd.DataFrame(fn_samples_dict)
            fn_samples.to_csv(os.path.join(results_testing_path, 'fn_samples.csv'), index=False)

            #   ---------------------------------------------------------------------------------------------

            #   TODO: Calculate the Fischer Linear Discriminant (FLD) from preds and trues

        if strategy == 0:
            preds = (preds > threshold_value).astype(float)
        elif strategy == 1:
            #   Decode predictions by taking argmax (vanilla multi-class..)
            preds = np.argmax(preds, axis=1)
        elif strategy == 2:
            #   TODO: convert torch tensor expectations for numpy

            #   1. Manually decode labels
            trues = enc.decode_labels(trues)

            #   2. Manually decode predictions
            preds = enc.decode_preds(preds)

        #   Calculate other metrics

        #   TODO: average parameter depends on strategy
        if strategy == 0:
            average = 'binary'
        else:
            average = 'macro'

        acc_score = accuracy_score(trues, preds)
        precision = precision_score(trues, preds, average=average)
        recall = recall_score(trues, preds, average=average)
        f1 = f1_score(trues, preds, average=average)

        conf_matrix = confusion_matrix(trues, preds)

        print(f'\nEvaluation of model (best epoch: {best_epoch}) on test split:\n')

        print(f"Accuracy: {round(acc_score, relevant_digits)}")
        print(f"Precision: {round(precision, relevant_digits)}")
        print(f"Recall: {round(recall, relevant_digits)}")
        print(f"F1-score: {round(f1, relevant_digits)}")
        print('\nConfusion matrix:')
        print(conf_matrix)

        # Save calculated metrics

        results_test_split = {
            'threshold_chosen': threshold_value,
            'accuracy': acc_score,
            'precision': precision,
            'recall': recall,
            'f1-score': f1
        }

        with open(os.path.join(results_testing_path, 'metrics.json'), 'w') as f:
            json.dump(results_test_split, f, indent=4)

        cm_display = ConfusionMatrixDisplay(conf_matrix)
        fig, ax = plt.subplots(figsize=(8, 8))
        cm_display.plot(ax=ax)
        ax.set_title(f"Confusion Matrix (threshold = {threshold_value}) - Evaluation on test data")
        plt.savefig(os.path.join(results_testing_path, 'confusion_matrix.png'), dpi=300)

    # ---------------------------------------------------------------------------------------------------------

    print(f'\nExperiment "{config_filename}": \n')

    #   Create Encoder instance

    enc = Encoder(strategy)

    #   Create data loader for train, validation and test subset

    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(target_input_height=input_height,
                                                                          target_input_width=input_width,
                                                                          interpolation_method=interpolation_method,
                                                                          enc=enc)

    #   Initialize model params and train model params with optimizer and loss function

    #   TODO -> Finetune ResNet

    if strategy == 0:
        num_out_features = 1
        outputs_function = "sigmoid"
        loss_fn = nn.BCELoss()
    elif strategy == 1:
        num_out_features = 3
        outputs_function = "softmax"
        loss_fn = nn.CrossEntropyLoss()
    elif strategy == 2:
        num_out_features = 7
        outputs_function = "sigmoid"
        loss_fn = nn.MSELoss(reduction='sum')
    else:
        raise Exception("Invalid strategy passed.")

    if model_name == 'custom':
        model = CustomModel2d(input_height=input_height, input_width=input_height)
        model.initialize_weights()
    elif model_name == 'resnet18':
        model = ResNet18(num_out_features=num_out_features,
                         outputs_function=outputs_function,
                         pretrained=pretrained)
    elif model_name == 'resnet34':
        model = ResNet34(num_out_features=num_out_features,
                         outputs_function=outputs_function,
                         pretrained=pretrained)
    else:
        raise Exception("Invalid model name passed.")

    #   Move model to device and set data type
    model = model.to(device=device, dtype=torch.float)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    model, best_epoch, best_epoch_weights_path = train_model(model,
                                                             num_epochs,
                                                             train_dataloader,
                                                             valid_dataloader,
                                                             optimizer,
                                                             loss_fn)

    #   Evaluate trained model (best epoch wrt. validation loss) on test data

    evaluate_on_test_data(model, best_epoch_weights_path, best_epoch, test_dataloader)

    print('-----------------------------------------------------------------------------------------')


if __name__ == '__main__':
    run_experiment(config=config)
