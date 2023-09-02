from common import os, sys, yaml, argparse, shutil, pd
from common import torch, nn, optim, sigmoid, softmax
from common import device

from model import CustomModel2d
from model import ResNet18
from model import ResNet34

from data import create_data_splits, get_dataloaders

from train import train_model

from evaluation import PerformanceEvaluator


def run_experiment(config: dict, experiment_name: str):

    # ---------------------------------------------------------------------------------------------------------

    #   Extract parameters from yaml config

    #   Create experiment results directory

    results_dir = os.path.join(os.getcwd(), 'results')

    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, experiment_name)

    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.makedirs(results_path)

    preds_dir = os.path.join(results_path, f'preds_{experiment_name}')

    os.makedirs(preds_dir, exist_ok=True)

    try:
        num_randomizations = config['num_randomizations']
    except KeyError as e:
        sys.exit(f'Error caught: Required key {e} could not be found in passed config.')

    # ---------------------------------------------------------------------------------------------------------

    #   Log the config file used for the experiment to results

    with open(os.path.join(results_path, 'config_used.yaml'), 'w') as f:
        yaml.dump(config, f)

    # ---------------------------------------------------------------------------------------------------------

    cwd_dir = os.getcwd()

    #   Load id-to-split table
    
    id_to_split_table_filepath = os.path.join(cwd_dir, '..', 'randomSplits03-Aug-2023-17-47-32.xlsx')

    id_to_split_table = pd.read_excel(id_to_split_table_filepath)

    ids = id_to_split_table['ID']

    id_to_split_table_cols = list(id_to_split_table.columns)
    id_to_split_table_cols.remove('ID')

    #   Choose first num_randomizations columns of table as split columns
    splits_for_randomization = id_to_split_table_cols[0:num_randomizations]

    print(f'\nExperiment "{experiment_name}" ({len(splits_for_randomization)} randomizations): \n')

    for i in splits_for_randomization:
        print("*" * 80)
        print(f'\n--- Randomization "{i}" ---\n')
        
        splits = id_to_split_table[i]

        id_to_split_dict = dict(zip(ids, splits))

        perform_experiment_given_randomization(randomization=i, config=config, 
                                               results_path=results_path, 
                                               preds_dir=preds_dir,
                                               id_to_split_dict=id_to_split_dict)
        
        print(f'\n--- Randomization "{i}" finished. ---\n')
        print("*" * 80)

    print(f'\nExperiment "{experiment_name}" finished. \n')

    print("-" * 160)


def perform_experiment_given_randomization(randomization: str, config, results_path, preds_dir, id_to_split_dict: dict):
    try:
        images_dirpath = config['images_dir']
        labels_filepath = config['labels_filepath']

        (input_height, input_width) = config['target_img_size']
        interpolation_method = config['interpolation_method']

        model_name = config['model_name']
        starting_weights_path = config['starting_weights_path']
        pretrained = config['pretrained']
        strategy = config['strategy']

        label_selection_strategy_train = config['label_selection_strategy_train']
        label_selection_strategy_valid = config['label_selection_strategy_valid']

        batch_size = config['batch_size']
        lr = config['lr']
        num_epochs = config['epochs']
    except KeyError as e:
        sys.exit(f'Error caught: Required key {e} could not be found in passed config.')

    cwd_dir = os.getcwd()

    #   Create temporary target data directory with required directory structure
    #    (exists only for the currently running experiment)
    target_data_dir = os.path.join(cwd_dir, 'data')

    if os.path.exists(target_data_dir) and os.path.isdir(target_data_dir):
        try:
            shutil.rmtree(target_data_dir)
        except OSError as e:
            print(f"Error: {target_data_dir} could not be removed; {e}")
    
    os.makedirs(target_data_dir)

    #   Create data folder containing the splits for given randomization

    create_data_splits(source_data_dir=images_dirpath, 
                       target_data_dir=target_data_dir, 
                       id_to_split_dict=id_to_split_dict)

    #   Create dataloader for train, validation and test split given dataset

    dataloaders = get_dataloaders(features_dirpath=target_data_dir,
                                  labels_filepath=labels_filepath,
                                  batch_size=batch_size,
                                  label_selection_strategy_train=label_selection_strategy_train,
                                  label_selection_strategy_valid=label_selection_strategy_valid,
                                  strategy=strategy,
                                  target_input_height=input_height,
                                  target_input_width=input_width,
                                  interpolation_method=interpolation_method)

    train_dataloader, valid_dataloader, test_dataloader = dataloaders

    #   Initialize model params and train model params with optimizer and loss function

    #   TODO -> Finetune ResNet


    #   TODO : samples welche für train_loss reduktion schwierig sind -> id's bestimmen

    if strategy == 'baseline':
        num_out_features = 1
        outputs_function = sigmoid
        loss_fn = nn.BCELoss()
    elif strategy == 'regression':
        num_out_features = 1
        outputs_function = sigmoid  # or None ?
        loss_fn = nn.MSELoss()
    elif strategy == 2:
        #   TODO: Achtung Baustelle..
        num_out_features = 7
        outputs_function = lambda x: softmax(x, dim=1)
        loss_fn = nn.MSELoss(reduction='sum')
    else:
        raise ValueError("Invalid value for config parameter strategy passed.")

    if model_name == 'custom':
        model = CustomModel2d(input_height=input_height, input_width=input_height)
        model.initialize_weights()
    elif model_name == 'resnet18':
        model = ResNet18(num_out_features=num_out_features,
                         outputs_activation_func=outputs_function,
                         pretrained=pretrained)
    elif model_name == 'resnet34':
        model = ResNet34(num_out_features=num_out_features,
                         outputs_activation_func=outputs_function,
                         pretrained=pretrained)
    else:
        raise Exception("Invalid model name passed.")

    #   Move model to device and set data type
    model = model.to(device=device, dtype=torch.float)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    if starting_weights_path is not None:
        model.load_state_dict(torch.load(starting_weights_path))

    #   The results path for the current randomization of image data
    results_path_for_randomization = os.path.join(results_path, 'randomization_'+str(randomization))

    model, best_epoch, model_weights_path = train_model(model,
                                                        num_epochs,
                                                        train_dataloader,
                                                        valid_dataloader,
                                                        optimizer,
                                                        loss_fn,
                                                        results_path_for_randomization)

    #   Evaluate trained model (best epoch wrt. validation loss) on test data

    performance_evaluator = PerformanceEvaluator(model=model,
                                                 model_weights_path=model_weights_path,
                                                 best_epoch=best_epoch,
                                                 test_dataloader=test_dataloader,
                                                 strategy=strategy,
                                                 results_path=results_path_for_randomization)
    performance_evaluator.evaluate_on_test_data()

    #   Concatenate predictions for train, validation and test cases

    preds_train_data = pd.read_csv(os.path.join(results_path_for_randomization, 'training', 'preds_train_data.csv'))
    preds_train_data['split'] = 1

    preds_valid_data = pd.read_csv(os.path.join(results_path_for_randomization, 'training', 'preds_valid_data.csv'))
    preds_valid_data['split'] = 2

    preds_test_data = pd.read_csv(os.path.join(results_path_for_randomization, 'testing', 'preds_test_data.csv'))
    preds_test_data['split'] = 3

    preds_all = pd.concat([preds_train_data, preds_valid_data, preds_test_data], ignore_index=True)
    
    preds_all.to_csv(os.path.join(preds_dir, f'preds_all_{str(randomization)}.csv'), index=False)

    #   Remove data splits related to current experiment
    
    if os.path.exists(target_data_dir) and os.path.isdir(target_data_dir):
        try:
            shutil.rmtree(target_data_dir)
        except OSError as e:
            print(f"Error: {target_data_dir} could not be removed; {e}")

if __name__ == '__main__':
    #   Parse command line arguments

    parser = argparse.ArgumentParser(description='Script for running experiments using a YAML configuration file.')

    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('-n', '--name', type=str, required=True, help='Name of experiment '
                                                                      '(also used as name for the results directory).')

    args = parser.parse_args()

    #   Load configuration yaml file into config dictionary

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config_filename = os.path.basename(args.config).removesuffix('.yaml')

    run_experiment(config=config, experiment_name=args.name)
