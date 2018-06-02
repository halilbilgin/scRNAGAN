from glob import glob
import numpy as np
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from libraries.analysis import Analysis
import csv
from libraries.IO import get_IO

def get_basename(path):
    return os.path.basename(os.path.normpath(path))

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_dir', type=str,
                    help='experiment directory')
parser.add_argument('--IO', type=str, default='npy',
                    help='type of experiment results')

args = parser.parse_args()

folders = glob(args.exp_dir+"/*/")
IO = get_IO(args.IO)
analysis_folder = args.exp_dir + '/analysis/'

if not os.path.exists(analysis_folder):
    os.makedirs(analysis_folder)
if not os.path.exists(analysis_folder+'pca_plots'):
    os.makedirs(analysis_folder + 'pca_plots')
if not os.path.exists(analysis_folder+'marker_plots'):
    os.makedirs(analysis_folder + 'marker_plots')

plt.ioff()
folders.sort()
results = []
for folder in folders:
    if get_basename(folder) == 'analysis':
        continue

    print("current exp:", folder)
    try:
        analysis = Analysis(folder, False, IO)

        epochs = [0, 10, 15, 20, 25, 30]
        index_scores = np.asarray([analysis.get_index_scores(analysis.get_generated_ratio(i)) for i in epochs])
        minIterIndex = np.argmin(np.mean(np.abs(index_scores), axis=1))

        print(('Min index', np.mean(np.abs(index_scores[minIterIndex]))))

        analysis.plot_pca(epochs[minIterIndex], analysis_folder+'pca_plots/'+get_basename(folder)+'.jpg')

        analysis.plot_ratios(epochs[0:minIterIndex+1], analysis_folder+'marker_plots/'+get_basename(folder)+'.jpg')

        row = {'exp_id': get_basename(folder), 'best_epoch': epochs[minIterIndex]}
        row = merge_two_dicts(row, analysis.get_hyperparams())

        row['data_path'] = get_basename(row['data_path'])

        for key in ['experiment_path', 'generator_output_activation', 'learning_schedule', 'seed', 'log_transformation']:
            row.pop(key, None)

        index_scores[minIterIndex] = np.abs(index_scores[minIterIndex])

        row['ind_mean'] = np.mean(np.abs(index_scores[minIterIndex]))

        results.append(row)

    except Exception as err:
        print(err)

results= sorted(results, key=lambda k: k['ind_mean'])

keys = results[0].keys()

with open(analysis_folder+'results.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results)

