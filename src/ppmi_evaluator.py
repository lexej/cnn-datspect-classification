from common import os, sys, yaml, shutil
from common import torch, DataLoader

from evaluation import _get_predictions, _save_preds

from data import PPMIDataset

def main(results_dir: str, features_normal_dir: str, features_reduced_dir: str):

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

    features_normal_dir = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/PPMI_dataset/2D/normal_HC'
    features_reduced_dir = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/PPMI_dataset/2D/reduced_PD'

    for i in results_dirs:
        main(results_dir=i, 
             features_normal_dir=features_normal_dir, 
             features_reduced_dir=features_reduced_dir)

