from common import os, nib, torch, sys, re, yaml, shutil
from common import sigmoid, Dataset, DataLoader

from data import preprocess_image

from evaluation import _get_predictions, _save_preds


#   Dataset only used for testing performance of models
class PPMIDataset(Dataset):
    def __init__(self, features_normal_dir, features_reduced_dir, 
                 target_input_height, target_input_width, interpolation_method):
        
        features_normal_filepaths = sorted(os.listdir(features_normal_dir))
        features_normal_filepaths = [os.path.join(features_normal_dir, x) for x in features_normal_filepaths]
        labels_normal = torch.zeros(len(features_normal_filepaths))

        features_reduced_filepaths = sorted(os.listdir(features_reduced_dir))
        features_reduced_filepaths = [os.path.join(features_reduced_dir, x) for x in features_reduced_filepaths]
        labels_reduced = torch.ones(len(features_reduced_filepaths))

        self.features_filepaths = features_normal_filepaths + features_reduced_filepaths
        self.labels = torch.cat((labels_normal, labels_reduced))

        self.target_input_height = target_input_height
        self.target_input_width = target_input_width
        self.interpolation_method = interpolation_method


    def __getitem__(self, index):

        target_filepath = self.features_filepaths[index]

        ######################################################################################################
        #   Get Image

        img = nib.load(target_filepath).get_fdata()

        img = preprocess_image(img, self.target_input_height, self.target_input_width, self.interpolation_method)
        
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



def main(results_dir):

    features_normal_dir = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/PPMI_dataset/2D/normal_HC'
    features_reduced_dir = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/PPMI_dataset/2D/reduced_PD'

    #   Create dir for preds for ppmi dataset

    ppmi_preds_path = os.path.join(results_dir, 'preds_ppmi')

    if os.path.exists(ppmi_preds_path):
        shutil.rmtree(ppmi_preds_path)
    os.makedirs(ppmi_preds_path)  

    #   Load config params needed for creating PPMIDataset

    with open(os.path.join(results_dir, 'config_used.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    (input_height, input_width) = config['target_img_size']
    interpolation_method = config['interpolation_method']
    batch_size = config['batch_size']

    ppmi_dataset= PPMIDataset(features_normal_dir=features_normal_dir, 
                                features_reduced_dir=features_reduced_dir, 
                                target_input_height=input_height,
                                target_input_width=input_width,
                                interpolation_method=interpolation_method)
    
    ppmi_dataloader = DataLoader(dataset=ppmi_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    results_dir_subdirs = os.listdir(results_dir)
    randomization_dirs = [dir_string for dir_string in results_dir_subdirs if dir_string.startswith('randomization')]


    for subdir in randomization_dirs:
        randomization_dir_path = os.path.join(results_dir, subdir)
        
        model_path = os.path.join(randomization_dir_path, 'training', 'model_with_best_weights.pth')

        model = torch.load(model_path)
        model.eval()

        ids_ppmi, preds_ppmi, labels_ppmi = _get_predictions(model, ppmi_dataloader)

        _save_preds(ids=ids_ppmi, preds=preds_ppmi, trues=labels_ppmi,
                    save_to=os.path.join(ppmi_preds_path, f'preds_ppmi_{os.path.basename(results_dir)}_{subdir}.csv'))


if __name__ == "__main__":
    # dir which contains the results for x-randomizations of the experiment given certain strategy
    results_dir_1 = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/calculated_results/results_pt1/baseline_majority'
    results_dir_2 = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/calculated_results/results_pt1/baseline_random'
    results_dir_3 = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/calculated_results/results_pt1/regression'

    results_dir_4 = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/calculated_results/results_pt2/baseline_majority'
    results_dir_5 = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/calculated_results/results_pt2/baseline_random'
    results_dir_6 = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/calculated_results/results_pt2/regression'

    results_dirs = [results_dir_1, results_dir_2, results_dir_3, results_dir_4, results_dir_5, results_dir_6]

    for i in results_dirs:
        main(results_dir=i)

