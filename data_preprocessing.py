import os
import sys
import shutil
import numpy as np

#   Script for creatig target data stucture from affine_2d folder
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

    for _id in subject_ids:
    
        subject_subdir = str(_id).zfill(num_digits)

        subject_subdir_full = os.path.join(target_data_dir, subject_subdir)

        os.makedirs(subject_subdir_full)

        #   Copy images from source_data_dir with current _id to target subject_subdir

        #   1. list of paths within source dir where filename end with _id

        matching_files = []
        for root, _, filenames in os.walk(source_data_dir):
            for filename in filenames:
                if filename.endswith(subject_subdir + FILE_FORMAT):
                    matching_files.append(os.path.join(root, filename))
        
        matching_files = np.array(matching_files)
        
        #   2. Copy all matching_files to subject_subdir_full

        for mf in matching_files:
            
            mf_relpath = os.path.relpath(path=mf, start=source_data_dir)
            
            target_file_name = mf_relpath.replace(os.path.sep, '_')
            
            shutil.copy(mf, os.path.join(subject_subdir_full, target_file_name))

    
    #   Split into training, testing and calibration set

    #   1. Create the folders 'training', 'testing', 'calibration'

    #  os.makedirs(target_data_dir)

    

    
    print('Target data created successfully.')


if __name__ == '__main__':
    main()
