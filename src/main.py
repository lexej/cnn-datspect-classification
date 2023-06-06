import os
import time
import subprocess


#   Path relative to current file
configs_dirpath = 'configs'

#   Filter out file name starting with '_'
configs_filepaths = list(filter(lambda x: not x.startswith('_'), os.listdir(configs_dirpath)))


if __name__ == '__main__':
    #   Launch the experiments in parallel

    start = time.time()

    #   Run experiment for each custom config existing in configs directory
    for f_name in configs_filepaths:
        filepath = os.path.join(configs_dirpath, f_name)
        if os.path.isfile(filepath):

            command = ["python", "experiment.py", "-c", filepath]

            subprocess.call(" ".join(command), shell=True)

    end = time.time()

    print(f'Experiments finished. Elapsed time: {round(end - start, 3)} sec')
