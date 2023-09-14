from common import os, sys, shutil, re, np, pd, nib
from common import torch, transforms, Dataset, DataLoader, InterpolationMode
from common import RANDOM_SEED, generator
from common import device
from common import train_test_split


def create_data_splits(source_data_dir, target_data_dir, id_to_split_dict: dict):

    print('--- Creating splits of image data ---')

    #   ID's of subjects
    subject_ids = np.arange(1, 1741)

    FILE_FORMAT = '.nii'

    num_digits = len(str(subject_ids[-1]))

    split_folder_names = {split_name for split_name in id_to_split_dict.values()}

    for sfn in split_folder_names:
        os.makedirs(os.path.join(target_data_dir, sfn))

    #   Copy data from source folder to target splits folders

    augmented_data_len = 0

    for _id in subject_ids:

        target_folder = os.path.join(target_data_dir, id_to_split_dict[_id])

        _id_filled = str(_id).zfill(num_digits)

        #   Get list of image paths of subject _id

        matching_files = []
        for root, _, filenames in os.walk(source_data_dir):
            for filename in filenames:
                if filename.endswith(_id_filled + FILE_FORMAT):
                    matching_files.append(os.path.join(root, filename))
        
        #   Copy all matching_files to target split folder

        for mf in matching_files:
            
            mf_relpath = os.path.relpath(path=mf, start=source_data_dir)
            
            target_file_name = mf_relpath.replace(os.path.sep, '_')
            
            shutil.copy(mf, os.path.join(target_folder, target_file_name))

            augmented_data_len += 1

    #   Check validity
    
    percentage_in_train = round(len(os.listdir(os.path.join(target_data_dir, 'train'))) / augmented_data_len, 2)
    percentage_in_test = round(len(os.listdir(os.path.join(target_data_dir, 'test'))) / augmented_data_len, 2)
    percentage_in_valid = round(len(os.listdir(os.path.join(target_data_dir, 'valid'))) / augmented_data_len, 2)

    if percentage_in_train != 0.60 or percentage_in_test != 0.20 or percentage_in_valid != 0.20:
        raise Exception('Invalid distribution of data to split folders.')

    print('--- Splits of image data successfully created. ---')


def preprocess_image(img: np.ndarray, resize: bool, 
                     target_input_height: int, target_input_width: int, interpolation_method: str, 
                     batch_dim: bool = False) -> torch.Tensor:

    #   1. Crop image of size (91, 109) to region of interest (91, 91)
    #       policy: select all rows and 10th to 100th column

    transformed_img = img[:, 9:100]

    #   2. Convert to torch tensor

    transforms_list = [
        transforms.ToTensor()
    ]

    if resize:
        #   3. If resize=True, resize images from (91, 91) to target size (input_height, input_width)
        #           using certain interpolation method
        #       - Attention: Interpolation method can have impact on model performance !
        resize = transforms.Resize(size=(target_input_height, target_input_width),
                          interpolation=getattr(InterpolationMode, interpolation_method.upper()),
                          antialias=True)
        transforms_list.append(resize)

    img_transforms = transforms.Compose(transforms_list)

    transformed_img = img_transforms(transformed_img)

    #   4. Convert to float type and move to device

    transformed_img = transformed_img.to(device=device, dtype=torch.float)

    if batch_dim:
        #   prepend batch dimension; proper working of layers which potentially require batch dimension e.g. BatchNorm
        transformed_img = torch.unsqueeze(transformed_img, 0)

    return transformed_img


class LabelSelector(Dataset):
    def __init__(self, subset: Dataset, label_function, label_selection_strategy: str, strategy: str):
        self.subset = subset

        self.label_function = label_function
        self.label_selection_strategy = label_selection_strategy
        self.strategy = strategy

    def __getitem__(self, index):

        img, available_labels, metadata = self.subset[index]

        #   Decide for label
        label = self.label_function(available_labels, self.label_selection_strategy, self.strategy)

        return img, label, metadata

    def __len__(self):
        return len(self.subset)


class SpectSubset(Dataset):
    """SpectSubset represents a split (train/validation/test) of the Spect dataset."""
    def __init__(self, features_path, labels_filepath, resize: bool, 
                 target_input_height, target_input_width, interpolation_method):
        self.features_path = features_path
        self.features_files = sorted(os.listdir(features_path))

        self.labels_filepath = labels_filepath
        self.labels = pd.read_excel(labels_filepath)

        self.resize = resize

        self.target_input_height = target_input_height
        self.target_input_width = target_input_width
        self.interpolation_method = interpolation_method

    def __getitem__(self, index):

        target_file = self.features_files[index]

        ######################################################################################################
        #   Get Image

        image_path = os.path.join(self.features_path, target_file)

        img = nib.load(image_path).get_fdata()

        img = preprocess_image(img, 
                               resize=self.resize,
                               target_input_height=self.target_input_height, 
                               target_input_width=self.target_input_width, 
                               interpolation_method=self.interpolation_method)
        
        ######################################################################################################
        #   Extract Id
        
        id_pattern_in_filename = r'[1-9]+[0-9]*.nii'

        id_string_matched = re.findall(id_pattern_in_filename, target_file)[0]
        id_ = int(id_string_matched.removesuffix('.nii'))

        metadata = {
            'id': id_
        }

        ######################################################################################################
        #   Get available labels (no decision for label this point)

        label_row = self.labels.loc[self.labels['ID'].astype(int) == id_]

        #   Convert pandas DataFrame row to pandas Series of labels of interest

        labels_final = label_row[['R1', 'R2', 'R3']]

        available_labels = labels_final.reset_index(drop=True).squeeze().values

        return img, available_labels, metadata

    def __len__(self):
        return len(self.features_files)

class PPMIDataset(Dataset):
    """The PPMI dataset is only used for evaluation of the trained models."""
    def __init__(self, features_normal_dir, features_reduced_dir, resize: bool,
                 target_input_height, target_input_width, interpolation_method):
        
        features_normal_filepaths = sorted(os.listdir(features_normal_dir))
        features_normal_filepaths = [os.path.join(features_normal_dir, x) for x in features_normal_filepaths]
        labels_normal = torch.zeros(len(features_normal_filepaths))

        features_reduced_filepaths = sorted(os.listdir(features_reduced_dir))
        features_reduced_filepaths = [os.path.join(features_reduced_dir, x) for x in features_reduced_filepaths]
        labels_reduced = torch.ones(len(features_reduced_filepaths))

        self.features_filepaths = features_normal_filepaths + features_reduced_filepaths
        self.labels = torch.cat((labels_normal, labels_reduced))

        self.resize = resize

        self.target_input_height = target_input_height
        self.target_input_width = target_input_width
        self.interpolation_method = interpolation_method


    def __getitem__(self, index):

        target_filepath = self.features_filepaths[index]

        ######################################################################################################
        #   Get Image

        img = nib.load(target_filepath).get_fdata()

        img = preprocess_image(img, 
                               resize=self.resize,
                               target_input_height=self.target_input_height, 
                               target_input_width=self.target_input_width, 
                               interpolation_method=self.interpolation_method)
        
        ######################################################################################################
        #   Extract Id
        
        id_pattern_in_filename = r'slab(.+).nii'

        id_ = re.findall(id_pattern_in_filename, os.path.basename(target_filepath))[0]

        metadata = {
            'id': id_
        }

        ######################################################################################################
        #   Get label

        label = self.labels[index]

        return img, label, metadata

    def __len__(self):
        return len(self.features_filepaths)

class MPHDataset(Dataset):
    """The MPH dataset is only used for evaluation of the trained models. 
    ATTENTION: MPH dataset contains 30 samples with nan-valued pixels (out of total n=640)
    Nan-values pixels are mapped to median values."""
    def __init__(self, mph_features_dir: str, mph_labels_filepath: str, resize: bool, 
                 target_input_height, target_input_width, interpolation_method):
        
        features_filepaths = sorted(os.listdir(mph_features_dir))
        self.features_filepaths = [os.path.join(mph_features_dir, x) for x in features_filepaths]

        self.labels_table = pd.read_excel(mph_labels_filepath)

        self.resize = resize

        self.target_input_height = target_input_height
        self.target_input_width = target_input_width
        self.interpolation_method = interpolation_method

    def __getitem__(self, index):

        target_filepath = self.features_filepaths[index]

        ######################################################################################################
        #   Extract Id from filename

        id_pattern_in_filename = r'slab0*([1-9]+[0-9]*).nii'

        id_ = int(re.findall(id_pattern_in_filename, os.path.basename(target_filepath))[0])

        metadata = {
            'id': id_
        }

        ######################################################################################################
        #   Get image

        img = nib.load(target_filepath).get_fdata()
        
        img = preprocess_image(img, 
                               resize=self.resize,
                               target_input_height=self.target_input_height, 
                               target_input_width=self.target_input_width, 
                               interpolation_method=self.interpolation_method)
        
        if torch.isnan(img).any():
            # Map nan-valued pixels to median of image (excluding the nans)

            nan_mask = torch.isnan(img)
            img[nan_mask] = torch.median(img[~nan_mask])

        ######################################################################################################
        #   Get label given the id

        label = int(self.labels_table[self.labels_table['ID'] == id_]['scoreConsensus'])

        return img, label, metadata

    def __len__(self):
        return len(self.features_filepaths)


def _choose_label_from_available_labels(label: np.ndarray, 
                                        label_selection_strategy: str, 
                                        strategy: str) -> torch.Tensor:

    available_labels = sorted(list(label))

    if strategy == 'baseline' or strategy == 'pca_rfc':
        if label_selection_strategy == 'random':
            chosen_label = int(np.random.choice(available_labels))
        elif label_selection_strategy == 'majority':
            chosen_label = int(np.bincount(available_labels).argmax())
        else:
            raise Exception("Invalid label_selection_strategy passed.")
    elif strategy == 'regression':
        labels_bincount = np.bincount(available_labels)
        if labels_bincount[0] == 3:
            chosen_label = float(0)
        elif labels_bincount[0] == 2:
            chosen_label = float(1.0/3.0)
        elif labels_bincount[0] == 1:
            chosen_label = float(2.0/3.0)
        elif labels_bincount[0] == 0:
            chosen_label = float(1)
    else:
        raise ValueError("Invalid value for config parameter 'strategy' was passed.")

    #   Wrap as torch tensor
    chosen_label = torch.tensor(data=[chosen_label],
                                dtype=torch.float32,
                                device=device)

    return chosen_label


def get_dataloaders(features_dirpath, 
                    labels_filepath,
                    batch_size,
                    label_selection_strategy_train,
                    label_selection_strategy_valid,
                    strategy: str,
                    resize: bool,
                    target_input_height,
                    target_input_width,
                    interpolation_method):

    # -----------------------------------------------------------------------------------------------------------

    #   Subset directory paths

    train_data_path = os.path.join(features_dirpath, 'train')

    valid_data_path = os.path.join(features_dirpath, 'valid')

    test_data_path = os.path.join(features_dirpath, 'test')

    #   Train set
    
    train_subset = SpectSubset(features_path=train_data_path,
                               labels_filepath=labels_filepath,
                               resize=resize,
                               target_input_height=target_input_height, 
                               target_input_width=target_input_width, 
                               interpolation_method=interpolation_method)
    
    train_subset = LabelSelector(subset=train_subset, 
                                 label_function=_choose_label_from_available_labels,
                                 label_selection_strategy=label_selection_strategy_train,
                                 strategy=strategy)

    #   Validation set

    val_subset = SpectSubset(features_path=valid_data_path,
                             labels_filepath=labels_filepath,
                             resize=resize,
                             target_input_height=target_input_height, 
                             target_input_width=target_input_width, 
                             interpolation_method=interpolation_method)
    
    val_subset = LabelSelector(subset=val_subset, 
                               label_function=_choose_label_from_available_labels,
                               label_selection_strategy=label_selection_strategy_valid,
                               strategy=strategy)
    
    #   Test set

    test_subset = SpectSubset(features_path=test_data_path,
                              labels_filepath=labels_filepath,
                              resize=resize,
                              target_input_height=target_input_height, 
                              target_input_width=target_input_width, 
                              interpolation_method=interpolation_method)

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
