from libraries.input_data import InputData, Scaling
from libraries.acgan import ACGAN
import json
import argparse
import os
import copy
from train import train

parser = argparse.ArgumentParser()

parser.add_argument("-epath", "--experiments_path",
                    help="path of the directory where experiment folder locates")
parser.add_argument("-rerun", "--rerun", type=int, default=0,
                    help="should it rerun the experiment if it is already run")
parser.add_argument("-rname", "--run_name", default="run1",
                    help="name of the directory where the results will be saved")
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
parser.add_argument("-d_steps", "--d_steps", default=1, type=int,
                    help="discriminator steps in each iteration")

args = parser.parse_args()

if not os.path.isdir(args.experiments_path):
    raise FileNotFoundError("Not correct path")

experiments_path = args.experiments_path
rerun = True if args.rerun == 1 else False
subdirectories = next(os.walk(experiments_path))[1]
subdirectories.sort()
del args.rerun, args.experiments_path
experiment_path = os.path.join(experiments_path, subdirectories[0])

for subdir in subdirectories:
    experiment_path = os.path.join(experiments_path, subdir)
    config_file = os.path.join(experiment_path, "config.json")
    if not os.path.isfile(config_file):
        continue
    if os.path.isdir(os.path.join(experiment_path, args.run_name)) and not rerun:
        continue

    clone_args = copy.deepcopy(args)
    clone_args.experiment_path = experiment_path

    train(clone_args)