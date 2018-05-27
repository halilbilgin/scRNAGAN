import argparse
import os
import copy
from train import train

parser = argparse.ArgumentParser()

parser.add_argument("-epath", "--experiments_path",
                    help="path of the directory where experiment folder locates")
parser.add_argument("-repeat", "--repeat", type=int, default=1,
                    help="the number of time an experiment will rerun")
parser.add_argument("-epochs", "--epochs", default = 25, type=int,
                    help="number of epochs")
parser.add_argument("-s_freq", "--summary_freq", type=int, default = 10,
                    help="tensorboard summary log frequency (iterations)")
parser.add_argument("-p_freq", "--print_freq", default = 10, type=int,
                    help="print frequency (iterations)")
parser.add_argument("-l_freq", "--log_sample_freq", default = 100, type=int,
                    help="generator sample log frequency (iterations)")
parser.add_argument("-l_size", "--log_sample_size", default = 250, type=int,
                    help="generator sample log frequency (iterations)")

args = parser.parse_args()

if not os.path.isdir(args.experiments_path):
    raise FileNotFoundError("Not correct path")

experiments_path = args.experiments_path
repeat = args.repeat
subdirectories = next(os.walk(experiments_path))[1]
subdirectories.sort()
del args.repeat, args.experiments_path
experiment_path = os.path.join(experiments_path, subdirectories[0])

for subdir in subdirectories:
    experiment_path = os.path.join(experiments_path, subdir)
    config_file = os.path.join(experiment_path, "config.json")
    if not os.path.isfile(config_file):
        continue
    for i in range(repeat):
        if os.path.isdir(os.path.join(experiment_path, 'run_' + str(repeat-1))):
            continue

        clone_args = copy.deepcopy(args)
        clone_args.experiment_path = experiment_path

        train(clone_args)