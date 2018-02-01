import sys
import argparse
import json 
import os
from libraries.utils import check_activation_function, check_scaling
from libraries.input_data import Scaling

def create_experiment(config):
    if config['scaling'] in Scaling.__members__:
        if Scaling[config['scaling']] == Scaling.minmax:
            config['generator_output_activation'] = 'sigmoid'
        elif Scaling[config['scaling']] == Scaling.standard:
            config['generator_output_activation'] = 'tanh'
    else:
        config['generator_output_activation'] = 'none'

    with open(config['experiment_path'] + '/config.json', 'w') as outfile:
        json.dump(config, outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-dpath", "--data_path", type=str)
    parser.add_argument("-epath", "--experiment_path", type=str)

    # python3 create_experiment.py -dpath ./Data/segerstolpe_healthy_250features -epath ./out/onemlibirdeney -d_hlayers '[180, 45]' -g_hlayers '[50, 200]' -bn 0 -log 1 -scaling minmax -act leaky_relu -out_act sigmoid -lr 4e-4 -z_dim 250 -mb_size 10 -g_drop 0.3 -d_drop 0.5 -label_noise 0.3

    parser.add_argument("-log", "--log_transformation", type=int)
    parser.add_argument("-scaling", "--scaling", type=check_scaling,
                       help="minmax = MinMaxScaler, standard = StandardScaler, none = no scaling")

    parser.add_argument("-d_hlayers", "--d_hidden_layers", type=json.loads,
                        help="discriminator hidden layers")

    parser.add_argument("-g_hlayers", "--g_hidden_layers", type=json.loads,
                        help="generator hidden layers")

    parser.add_argument("-bn", "--normalizer_fn", type=int,
                        help="use batch normalization")

    parser.add_argument("-act", "--activation_function", type=check_activation_function,
                        help="activation function of hidden layers")

    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="learning rate of Adam optimizer")

    parser.add_argument("-z_dim", "--z_dim", type=int,
                        help="generator noise input dimension")

    parser.add_argument("-mb_size", "--mb_size", type=int,
                        help="mini batch size")

    parser.add_argument("-g_drop", "--g_dropout", type=float,
                        help="generator dropout")

    parser.add_argument("-d_drop", "--d_dropout", type=float,
                        help="discriminator dropout")

    parser.add_argument("-label_noise", "--label_noise", type=float,
                        help="discriminator dropout")

    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print("Data path should exist.")
        sys.exit(0)

    if os.path.isdir(args.experiment_path):
        answer = ''
        while answer not in ['y', 'n']:
            answer = input("The dir you want to create experiment in already exists. Do you want to continue?(y/n)")
        if answer=='n':
            sys.exit(0)
    else:
        os.makedirs(args.experiment_path)

    config = vars(args)

    create_experiment(config)
