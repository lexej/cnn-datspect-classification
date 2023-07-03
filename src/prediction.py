import os
import argparse
import sys
import nibabel as nib

import torch
import yaml
from data import preprocess_image


#   This module predicts the diagnosis given a file path to a SPECT image in nii-format
def predict_with_confidence_estimation(experiment_name: str, test_case_path: str):

    results_dir = os.path.join(os.getcwd(), 'results')
    experiment_dir = os.path.join(results_dir, experiment_name)

    if not os.path.exists(experiment_dir):
        sys.exit(f"The experiment results directory '{experiment_dir}' does not exists.")

    #   Hyperparameters used for model training

    with open(os.path.join(experiment_dir, 'config_used.yaml'), 'r') as f:
        config_used = yaml.safe_load(f)

    #   Load test image and preprocess correctly

    img = nib.load(test_case_path).get_fdata()

    target_input_height, target_input_width = config_used['target_img_size']
    interpolation_method = config_used['interpolation_method']

    img = preprocess_image(img, target_input_height, target_input_width, interpolation_method, batch_dim=True)

    #   Model

    model_path = os.path.join(experiment_dir, 'training', 'model_with_best_weights.pth')

    model = torch.load(model_path)

    #   Prediction

    pred = model(img)

    print(f'\nPrediction: ', float(pred), f'\n for image {test_case_path} ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for prediction of diagnosis using the model from a certain '
                                                 'experiment.')

    parser.add_argument('-n', '--name', type=str, required=True, help='Name of experiment '
                                                                      '(also used as name for the results directory).')
    parser.add_argument('-fp', '--filepath', type=str, required=False, help='Path to the nii-file for prediction.')

    args = parser.parse_args()

    experiment_name = args.name

    #   Path to file of .nii SPECT image for prediction
    image_filepath = args.filepath

    if image_filepath:
        test_case_path = image_filepath
    else:
        #   default

        images_dir = '/Users/aleksej/IdeaProjects/master-thesis-kucerenko/UKE_SPECT/affine_p75_withACSC_multipleTemplates_woCerebellum/2d'

        file_name = 'slab0483.nii'

        test_case_path = os.path.join(images_dir, file_name)

    predict_with_confidence_estimation(experiment_name, test_case_path=test_case_path)
