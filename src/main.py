from experiment import run_experiment
import yaml
import time


#   Path relative to current file
configs_dirpath = 'configs'

#   Base config (can be overriden by custom config!)

base_config_path = f'{configs_dirpath}/_base_config.yaml'

#   Config for experiment for rigid case and 2D images:

with open(base_config_path, 'r') as f_base, open(f'{configs_dirpath}/config_rigid_2d.yaml', 'r') as f_custom:
    config_rigid_2d_config = yaml.safe_load(f_base)
    #   Update with custom changes
    config_rigid_2d_config.update(yaml.safe_load(f_custom))

#   Config for experiment for affine case and 2D images:

with open(base_config_path, 'r') as f_base, open(f'{configs_dirpath}/config_affine_2d.yaml', 'r') as f_custom:
    config_affine_2d_config = yaml.safe_load(f_base)
    #   Update with custom changes
    config_affine_2d_config.update(yaml.safe_load(f_custom))


if __name__ == '__main__':
    #   Launch the experiments in parallel

    start = time.time()

    run_experiment(config=config_rigid_2d_config)

    run_experiment(config=config_affine_2d_config)

    end = time.time()

    print(f'Experiments finished. Elapsed time: {round(end - start, 3)} sec')
