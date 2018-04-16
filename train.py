from libraries.input_data import InputData, Scaling
from libraries.acgan import ACGAN
import json
import argparse
import os
import sys
from libraries.utils import get_acgan
import tensorflow as tf
from libraries.IO import get_IO

# python3 train.py -epath /home/halilbilgin/remoteSeqGAN/out/experiment2 -i 3000 -s_freq 10 -p_freq 10 -l_freq 100 -l_size 250

def train(args, return_output=False):

    if not os.path.isdir(args.experiment_path):
        print("Experiment directory should exist.")
        sys.exit(0)

    if not os.path.isfile(args.experiment_path + '/config.json'):
        print("Model config file should exist.")
        sys.exit(0)

    dir_name = args.run_name
    i = 0
    while os.path.isdir(args.experiment_path + '/' + dir_name):
        dir_name = args.run_name + '_' + str(i)
        i += 1

    dir_name = args.experiment_path + '/' + dir_name

    with open(args.experiment_path + '/config.json') as json_file:
        config = json.load(json_file)

    config['experiment_path'] = args.experiment_path

    acgan = get_acgan(config)

    acgan.build_model()

    train_config = vars(args)
    _, train_labels = acgan.input_data.get_data()

    iters_per_epoch = int(train_labels.shape[0] / config['mb_size'] + 1)
    train_config['iterations'] = iters_per_epoch * train_config['epochs'] + 1

    del train_config['experiment_path'], train_config['run_name'], train_config['epochs'], train_config['IO']

    acgan.train_and_log(dir_name, acgan.input_data.IO, **train_config)
    if return_output:
        return acgan
    else:
        acgan.close_session()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-epath", "--experiment_path",
                        help="path of the directory where experiment config and results will be stored")
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

    args = parser.parse_args()

    train(args)