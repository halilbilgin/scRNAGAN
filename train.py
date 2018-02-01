from libraries.input_data import InputData, Scaling
from libraries.acgan import ACGAN
import json
import argparse
import os
import sys
from libraries.utils import get_activation
import tensorflow as tf

# python3 train.py -epath /home/halilbilgin/remoteSeqGAN/out/experiment2 -i 3000 -s_freq 10 -p_freq 10 -l_freq 100 -l_size 250 -d_steps 1

def train(args):
    print(args.experiment_path)

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

    input_data = InputData(config['data_path'])

    input_data.preprocessing(config['log_transformation'], None if config['scaling'] not in Scaling.__members__ else Scaling[config['scaling']])

    train_data, train_labels = input_data.get_data()
    if 'leaky_param' not in config:
        config['leaky_param'] = 0.1

    config['activation_function'] = get_activation(config['activation_function'], config['leaky_param'])
    config['generator_output_activation'] = get_activation('sigmoid' if config['scaling'] else 'none')



    if config['normalizer_fn'] != 1:
        config['normalizer_fn'] = None
    else:
        config['normalizer_fn'] = tf.contrib.layers.batch_norm
        config['normalizer_params'] = {'center': True, 'scale': True}

    acgan = ACGAN(train_data.shape[1], train_labels.shape[1], input_data, **config)

    acgan.build_model()

    train_config = vars(args)

    iters_per_epoch = int(train_data.shape[0] / config['mb_size'] + 1)
    train_config['iterations'] = iters_per_epoch * train_config['epochs'] + 1

    del train_config['experiment_path'], train_config['run_name'], train_config['epochs']

    acgan.train_and_log(input_data.iterator, dir_name, **train_config)
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
    parser.add_argument("-d_steps", "--d_steps", default=1, type=int,
                        help="discriminator steps in each iteration")

    args = parser.parse_args()

    train(args)