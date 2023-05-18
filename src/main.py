from experiment import run_experiment
import yaml


#   Path relative to current file
configs_dirpath = 'configs'

#   Run experiment for rigid case and 2D images:

config_rigid_2d_path = f'{configs_dirpath}/config_rigid_2d.yaml'

with open(config_rigid_2d_path, 'r') as f:
    config_rigid_2d = yaml.safe_load(f)

run_experiment(experiment_name='rigid_2d',
               config=config_rigid_2d)

#   Run experiment for affine case and 2D images:

config_affine_2d_path = f'{configs_dirpath}/config_affine_2d.yaml'

with open(config_affine_2d_path, 'r') as f:
    config_affine_2d = yaml.safe_load(f)

run_experiment(experiment_name='affine_2d',
               config=config_affine_2d)