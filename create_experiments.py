import argparse
from create_experiment import create_experiment
import json
import os
import sys
import copy
import random
import string


# Python program to print all paths from a source to destination.

def create_experiments(experiments_path, prefix, config):
    keys = list(config.keys())

    paths = []
    path = copy.deepcopy(config)

    def traverse_config(data, pathLen, curPath, paths):
        curPath[keys[pathLen]] = data

        if (pathLen == len(keys) - 1):
            paths.append(dict(curPath))
            curPath = dict(curPath)

            return

        pathLen += 1

        for i in config[keys[pathLen]]:
            traverse_config(i, pathLen, curPath, paths)

    traverse_config(config['data_path'][0], -1, path, paths)

    if not os.path.isdir(experiments_path):
        os.makedirs(experiments_path)

    seed = 23
    for cfg in paths:
        hash = ''.join(random.choice(string.ascii_uppercase) for _ in range(10))
        cfg['experiment_path'] = os.path.join(experiments_path,
                                              prefix + '_' + hash)
        cfg['seed'] = seed
        if not os.path.isdir(cfg['experiment_path']):
            os.makedirs(cfg['experiment_path'])

        create_experiment(cfg)

    f = open(os.path.join(experiments_path, prefix+'.json'), 'w')
    f.write(json.dumps(config))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-epath", "--experiments_path",
                        help="path of the directory where experiment folder locates")
    parser.add_argument("-cfg", "--config_file",
                        help="config file for creating experiments")

    args = parser.parse_args()

    if not os.path.isfile(args.config_file):
        print("Config file does not exist")
        sys.exit(0)

    with open(args.config_file) as json_file:
        config = json.load(json_file)

    experiments_path = args.experiments_path
    prefix = config['experiments_prefix']

    del config['experiments_prefix']

    create_experiments(experiments_path, prefix, config)
