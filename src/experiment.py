from common import os, yaml, argparse, shutil
from common import torch, nn, optim
from common import device

from model import CustomModel2d
from model import ResNet18
from model import ResNet34

from data import SpectDataset, get_dataloaders

from train import train_model

from evaluation import evaluate_on_test_data


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

    test_to_train_split_size_percent = config['test_to_train_split_size_percent']

    valid_to_train_split_size_percent = config['valid_to_train_split_size_percent']

    # ---------------------------------------------------------------------------------------------------------

    #   Log the config file used for the experiment to results

    with open(os.path.join(results_path, 'config_used.yaml'), 'w') as f:
        yaml.dump(config, f)

    # ---------------------------------------------------------------------------------------------------------

    print(f'\nExperiment "{experiment_name}": \n')

    #   Create SPECT dataset instance

    spect_dataset = SpectDataset(features_dirpath=images_dirpath,
                                 labels_filepath=labels_filepath,
                                 target_input_height=input_height,
                                 target_input_width=input_width,
                                 interpolation_method=interpolation_method)

    #   Create dataloader for train, validation and test split given dataset

    dataloaders = get_dataloaders(dataset=spect_dataset,
                                  batch_size=batch_size,
                                  test_to_train_split_size_percent=test_to_train_split_size_percent,
                                  valid_to_train_split_size_percent=valid_to_train_split_size_percent,
                                  label_selection_strategy_train=label_selection_strategy_train,
                                  label_selection_strategy_valid=label_selection_strategy_valid)

    train_dataloader, valid_dataloader, test_dataloader = dataloaders

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

    if starting_weights_path is not None:
        model.load_state_dict(torch.load(starting_weights_path))

    model, best_epoch, best_epoch_weights_path = train_model(model,
                                                             num_epochs,
                                                             train_dataloader,
                                                             valid_dataloader,
                                                             optimizer,
                                                             loss_fn,
                                                             results_path)

    #   Evaluate trained model (best epoch wrt. validation loss) on test data

    evaluate_on_test_data(model, best_epoch_weights_path, best_epoch, test_dataloader, strategy, results_path)

    print('-----------------------------------------------------------------------------------------')


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
