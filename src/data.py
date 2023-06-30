from common import os, np, pd, nib
from common import torch, transforms, Dataset, Subset, DataLoader, InterpolationMode
from common import RANDOM_SEED, generator
from common import device
from common import train_test_split


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

        id_ = idx + 1

        label_row = self.labels.loc[self.labels['ID'].astype(int) == id_]

        #   Convert pandas DataFrame row to pandas Series of labels of interest

        labels_full = label_row[['R1', 'R2', 'R3',
                                 'R1S1', 'R1S2', 'R2S1',
                                 'R2S2', 'R3S1', 'R3S2']].reset_index(drop=True).squeeze()

        #   Attention: For the general Dataset the label is a dictionary containing all available labels;
        #              the decision for the label is performed within Dataset classes wrapping train and valid Subset's
        #              of the current Dataset

        label = labels_full.to_dict()

        metadata = {
            'id': id_
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


class LabelFunctionWrapper(Dataset):
    def __init__(self, original_dataset: Dataset, label_function, label_selection_strategy: str):
        self.original_dataset = original_dataset
        self.label_function = label_function
        self.label_selection_strategy = label_selection_strategy

    def __getitem__(self, index):
        img, label, metadata = self.original_dataset[index]
        modified_label = self.label_function(label, self.label_selection_strategy)
        return img, modified_label, metadata

    def __len__(self):
        return len(self.original_dataset)


def _choose_label_from_available_labels(label: dict, label_selection_strategy: str) -> torch.Tensor:

    intra_rater_consensus_labels = {key: label[key] for key in ['R1', 'R2', 'R3']}

    available_labels = sorted(list(intra_rater_consensus_labels.values()))

    if label_selection_strategy == 'random':
        chosen_label = np.random.choice(available_labels)
    elif label_selection_strategy == 'majority':
        chosen_label = np.bincount(available_labels).argmax()
    else:
        raise Exception("Invalid label_selection_strategy passed.")

    #   Wrap as torch tensor
    chosen_label = torch.tensor(data=[int(chosen_label)],
                                dtype=torch.float32,
                                device=device)

    return chosen_label


def get_dataloaders(dataset: Dataset,
                    batch_size,
                    test_to_train_split_size_percent,
                    valid_to_train_split_size_percent,
                    label_selection_strategy_train,
                    label_selection_strategy_valid):

    # -----------------------------------------------------------------------------------------------------------

    #   Create data split for training, validation and testing

    #   1. train-test split  (no stratify used)

    train_indices, test_indices = train_test_split(range(len(dataset)),
                                                   test_size=test_to_train_split_size_percent,
                                                   random_state=RANDOM_SEED,
                                                   shuffle=True)

    train_subset = Subset(dataset=dataset, indices=train_indices)
    test_subset = Subset(dataset=dataset, indices=test_indices)

    #  2. test-validation split  (no stratify used)

    train_indices, val_indices = train_test_split(range(len(train_subset)),
                                                  test_size=valid_to_train_split_size_percent,
                                                  random_state=RANDOM_SEED,
                                                  shuffle=True)

    #   First create validation subset, then reassign train subset !

    val_subset = Subset(dataset=train_subset, indices=val_indices)

    train_subset = Subset(dataset=train_subset, indices=train_indices)

    #   train and valid Subsets are wrapped into a Dataset
    #   which applies a mapping function to the labels using a certain label selection strategy

    train_subset = LabelFunctionWrapper(original_dataset=train_subset,
                                        label_function=_choose_label_from_available_labels,
                                        label_selection_strategy=label_selection_strategy_train)

    val_subset = LabelFunctionWrapper(original_dataset=val_subset,
                                      label_function=_choose_label_from_available_labels,
                                      label_selection_strategy=label_selection_strategy_valid)

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

class Encoder:

    def __init__(self, strategy: int):
        self.strategy = strategy

    def get_label_using_strategy(self, final_labels: pd.Series, session_labels: pd.Series):

        final_labels_occurrences = final_labels.value_counts()
        session_labels_occurrences = session_labels.value_counts()

        count_of_0_in_final_labels = final_labels_occurrences.get(0, default=0)
        count_of_0_in_session_labels = session_labels_occurrences.get(0, default=0)
        count_of_1_in_session_labels = session_labels_occurrences.get(1, default=0)

        if self.strategy == 1:
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

        if self.strategy == 1:
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

"""

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
