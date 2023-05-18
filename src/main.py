from experiment import run_experiment
import yaml


#   Path relative to current file
configs_dirpath = 'configs'

#   Base config (can be overriden by custom config!)

base_config_path = f'{configs_dirpath}/base_config.yaml'

#   Run experiment for rigid case and 2D images:

with open(base_config_path, 'r') as f_base, open(f'{configs_dirpath}/config_rigid_2d.yaml', 'r') as f_custom:
    config_rigid_2d_config = yaml.safe_load(f_base)
    #   Update with custom changes
    config_rigid_2d_config.update(yaml.safe_load(f_custom))

run_experiment(experiment_name='rigid_2d',
               config=config_rigid_2d_config)

#   Run experiment for affine case and 2D images:

with open(base_config_path, 'r') as f_base, open(f'{configs_dirpath}/config_affine_2d.yaml', 'r') as f_custom:
    config_affine_2d_config = yaml.safe_load(f_base)
    #   Update with custom changes
    config_affine_2d_config.update(yaml.safe_load(f_custom))

run_experiment(experiment_name='affine_2d',
               config=config_affine_2d_config)
