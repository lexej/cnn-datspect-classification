import os
from experiment import run_experiment
import yaml
import time


#   Path relative to current file
configs_dirpath = 'configs'

#   Base config (can be overriden by custom config!)

base_config_path = f'{configs_dirpath}/_base_config.yaml'

configs_filepaths = list(filter(lambda x: not x.startswith('_'), os.listdir(configs_dirpath)))


if __name__ == '__main__':
    #   Launch the experiments in parallel

    start = time.time()

    #   Run experiment for each custom config existing in configs directory
    for f_name in configs_filepaths:
        filepath = os.path.join(configs_dirpath, f_name)
        if os.path.isfile(filepath):

            with open(base_config_path, 'r') as f_base, open(filepath, 'r') as f_custom:
                config = yaml.safe_load(f_base)
                #   Update with custom changes
                config.update(yaml.safe_load(f_custom))

                run_experiment(config=config)

    end = time.time()

    print(f'Experiments finished. Elapsed time: {round(end - start, 3)} sec')
