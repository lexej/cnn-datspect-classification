import os
import sys
import shutil
import numpy as np
import pandas as pd


#   Script for creating target data directory stucture from affine_2d directory

def main():
    print('\n')

    cwd_dir = os.getcwd()

    source_data_dir = os.path.join(cwd_dir, 'affine_2d')

    #   ID's of subjects
    subject_ids = np.arange(1, 1741)

    FILE_FORMAT = '.nii'

    #   Create target data folder
    target_data_dir = os.path.join(cwd_dir, 'data')

    if os.path.exists(target_data_dir) and os.path.isdir(target_data_dir):
        try:
            shutil.rmtree(target_data_dir)
        except OSError as e:
            print(f"Error: {target_data_dir} could not be removed; {e}")
    
    os.makedirs(target_data_dir)

    num_digits = len(str(subject_ids[-1]))

    #   Load id-to-split table and create subfolders for train, validation, test

    id_to_split_table = pd.read_excel(os.path.join(cwd_dir, 'randomSplits03-Aug-2023-17-47-32.xlsx'))
    
    id_to_split_dict = dict(zip(id_to_split_table['ID'], id_to_split_table['splittrain']))

    split_folder_names = list(id_to_split_table['splittrain'].unique())

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

    print('Target data directory prepared successfully.')


if __name__ == '__main__':
    main()
