#!/usr/bin/python -u

#SBATCH --output=/vol1/ibrahim/gan/out2.txt
#SBATCH --cpus-per-task=5# 3 cores
#SBATCH --job-name=architecture
#SBATCH --time=0
#SBATCH --nodes=1
#SBATCH --gres=gpu:0

import os
import sys
sys.path.append('/vol1/ibrahim/gan/')
from create_experiments import create_experiments
from train import train
import copy
import tensorflow as tf
import argparse

config = {

    "data_path": ["/vol1/ibrahim/data/alphabeta_joint_500/"],
    "log_transformation": [0],

    "scaling": ["none", "minmax"],

    "d_hidden_layers":[[600, 170]],

    "g_hidden_layers": [[144, 576]],

    "activation_function": ["relu", "leaky_relu", "tanh"],
    "leaky_param": [0.1],
    "learning_rate": [0.00001],
    "learning_schedule": ['no_schedule'],
    "optimizer": ['RMSProp'],
    "wgan": [0],
    "z_dim": [100],
    "mb_size": [1],
    "d_dropout":[0], "g_dropout": [0], "label_noise": [0]
}
# exp_dir = '/home/ibrahim/out/trial_tanh_g_dropout'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_dir', type=str,
                    help='experiment directory')
parser.add_argument('--create', type=int, default=1, help='create an experiment or use the existing')

args = parser.parse_args()
exp_dir = args.exp_dir
if (args.create):
    create_experiments(exp_dir, 'h_bn', config)

C = type('C', (object,), {})
args = C()
args.IO = 'npy'
args.epochs = 30
args.log_sample_freq = 5
args.log_sample_size = 500
args.summary_freq = 9999999
args.print_freq = 200
args.experiment_path = exp_dir
repeat = 4

subdirectories = next(os.walk(exp_dir))[1]
subdirectories.sort()

for subdir in subdirectories:
    experiment_path = os.path.join(exp_dir, subdir)
    config_file = os.path.join(experiment_path, "config.json")

    if not os.path.isfile(config_file):
        continue

    for i in range(repeat):

        if os.path.isdir(os.path.join(experiment_path, 'run_' + str(repeat - 1))):
            continue
        print(experiment_path, i)

        clone_args = copy.deepcopy(args)
        clone_args.experiment_path = experiment_path

        #config = tf.ConfigProto(device_count={'CPU': 50})
        sess = tf.Session()

        try:
            train(clone_args, sess=sess)
        except Exception as err:
            print(err)
