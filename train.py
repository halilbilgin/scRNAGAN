from libraries.acgan import ACGAN
import json
import argparse
import os
import sys
import tensorflow as tf
from libraries.utils import close_session

def train(args, return_output=False, sess = False):

    if not os.path.isdir(args.experiment_path):
        print("Experiment directory should exist.")
        sys.exit(0)

    if not os.path.isfile(args.experiment_path + '/config.json'):
        print("Model config file should exist.")
        sys.exit(0)

    with open(args.experiment_path + '/config.json') as json_file:
        config = json.load(json_file)

    dir_name = 'run_0'
    i = 1

    if 'seed' not in config:
        config['seed'] = 23

    while os.path.isdir(args.experiment_path + '/' + dir_name):
        dir_name = 'run_' + str(i)
        i += 1
        config['seed'] += 1

    dir_name = args.experiment_path + '/' + dir_name

    config['experiment_path'] = args.experiment_path
    if sess == False:
        sess = tf.Session()

    acgan, input_data = ACGAN.load(sess, config)

    train_config = vars(args)
    _, train_labels = input_data.get_data()

    iters_per_epoch = int(train_labels.shape[0] / config['mb_size'] + 1)
    train_config['iterations'] = iters_per_epoch * train_config['epochs'] + 1
    train_config['log_sample_freq'] = iters_per_epoch * train_config['log_sample_freq']

    del train_config['experiment_path'], train_config['epochs']
    if 'IO' in train_config:
        del train_config['IO']

    acgan.train_and_log(dir_name, input_data.IO, input_data, **train_config)

    if return_output:
        return acgan, dir_name
    else:
        acgan.save_session(dir_name)
        close_session(sess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-epath", "--experiment_path",
                        help="path of the directory where experiment config and results will be stored")
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