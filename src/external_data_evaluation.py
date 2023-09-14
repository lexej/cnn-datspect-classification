from common import os, sys, yaml, shutil, np, json
from common import torch, DataLoader, Dataset
from common import accuracy_score

from evaluation import _get_predictions, _save_preds

from pca_rfc import evaluate_rfc

from data import PPMIDataset, MPHDataset


def evaluate_on_external_dataset(results_dir: str, dataset: Dataset, batch_size, strategy, target_dir_name: str):

    print(f'--- Evaluation of "{os.path.basename(results_dir)}" on "{target_dir_name}" started.---')

    #   Create dir for preds and performance metrics on external dataset

    external_data_preds_path = os.path.join(results_dir, target_dir_name)

    if os.path.exists(external_data_preds_path):
        shutil.rmtree(external_data_preds_path)
    os.makedirs(external_data_preds_path)  
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False)

    results_dir_subdirs = os.listdir(results_dir)
    randomization_dirs = [dir_string for dir_string in results_dir_subdirs if dir_string.startswith('randomization')]


    for subdir in randomization_dirs:
        print(f'   ... randomization "{subdir}"')

        randomization_dir_path = os.path.join(results_dir, subdir)

        preds_save_to_path = os.path.join(external_data_preds_path, f'preds_{os.path.basename(results_dir)}_{subdir}.csv')
        
        if strategy == 'baseline' or strategy == 'regression':

            model_path = os.path.join(randomization_dir_path, 'training', 'model_with_best_weights.pth')

            model = torch.load(model_path)
            model.eval()

            ids, preds, labels = _get_predictions(model, dataloader)

            #   Save preds 

            _save_preds(ids=ids, preds=preds, trues=labels, save_to=preds_save_to_path)
            
            #   Save performance stats

            preds_thresholded = np.where(np.array(preds) >= 0.5, 1, 0)

            performance_stats = {
                'acc_score': accuracy_score(y_true=labels, y_pred=preds_thresholded)
            }

            performance_stats_path = os.path.join(external_data_preds_path, f'performance_{os.path.basename(results_dir)}_{subdir}.json')
            
            with open(performance_stats_path, 'w') as f:
                json.dump(performance_stats, f)

        elif strategy == 'pca_rfc':
            rfc_path = os.path.join(randomization_dir_path, 'training', 'rfc_trained.pkl')
            pca_path = os.path.join(randomization_dir_path, 'training', 'pca_trained.pkl')

            evaluate_rfc(rfc_path=rfc_path, 
                         pca_path=pca_path, 
                         test_dataloader=dataloader,
                         strategy=strategy,
                         save_to_path=preds_save_to_path)
    
    print(f'--- Evaluation of "{os.path.basename(results_dir)}" on "{target_dir_name}" finished.---')


if __name__ == "__main__":
    # dir which contains the results for x-randomizations of the experiment given certain strategy
    results_dir_1 = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_majority'
    results_dir_2 = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/baseline_random'
    results_dir_3 = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/regression'
    results_dir_4 = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/pca_rfc'

    results_dirs = [results_dir_1, results_dir_2, results_dir_3, results_dir_4]

    #   Paths to PPMI data

    ppmi_features_normal_dir = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/PPMI_dataset/2D/normal_HC'
    ppmi_features_reduced_dir = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/PPMI_dataset/2D/reduced_PD'

    #   Paths to MPH data

    mph_features_dir = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/MPH_30min_20230824/MPH_30min_20240824'
    mph_labels_filepath = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/MPH_30min_20230824/MPH_30min_20230824.xlsx'


    for i in results_dirs:
        #   -----------------------------------------------------------------------------

        #   Load config params needed for creating external dataset

        with open(os.path.join(i, 'config_used.yaml'), 'r') as f:
            config = yaml.safe_load(f)

        batch_size = config['batch_size']
        strategy = config['strategy']

        (input_height, input_width) = (None, None)
        interpolation_method = None

        if strategy == 'baseline' or strategy == 'regression':
            resize = True
            (input_height, input_width) = config['target_img_size']
            interpolation_method = config['interpolation_method']
        elif strategy == 'pca_rfc':
            resize = False

        #   -----------------------------------------------------------------------------
        #   1. Evaluate on PPMI dataset

        ppmi_dataset= PPMIDataset(features_normal_dir=ppmi_features_normal_dir, 
                                features_reduced_dir=ppmi_features_reduced_dir, 
                                resize=resize,
                                target_input_height=input_height,
                                target_input_width=input_width,
                                interpolation_method=interpolation_method)

        evaluate_on_external_dataset(results_dir=i, 
                                     dataset=ppmi_dataset, 
                                     batch_size=batch_size, 
                                     strategy=strategy,
                                     target_dir_name='preds_ppmi')


        #   -----------------------------------------------------------------------------
        #   2. Evaluate on MPH dataset
        
        mph_dataset = MPHDataset(mph_features_dir=mph_features_dir,
                                 mph_labels_filepath=mph_labels_filepath,
                                 resize=resize,
                                 target_input_height=input_height,
                                target_input_width=input_width,
                                interpolation_method=interpolation_method)

        evaluate_on_external_dataset(results_dir=i, 
                                     dataset=mph_dataset, 
                                     batch_size=batch_size, 
                                     strategy=strategy,
                                     target_dir_name='preds_mph')

        #   -----------------------------------------------------------------------------

