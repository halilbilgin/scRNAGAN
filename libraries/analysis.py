from sklearn import decomposition
from sklearn import preprocessing
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
from libraries.input_data import InputData, Scaling

class Analysis(object):

    def merge_train_and_generated_data(self, generated_data):
        train_data , _ = self.input_data.get_raw_data()

        data = np.concatenate((train_data, generated_data), axis=0)
        scaler = preprocessing.StandardScaler().fit(train_data)
        data[0:train_data.shape[0], :] = scaler.transform(train_data)
        data[train_data.shape[0]:, :] = scaler.transform(generated_data)


        return data, train_data.shape

    def tSNE(self, generated_data, filename, **kwargs):
        data, train_shape = self.merge_train_and_generated_data(generated_data)

        tsne = TSNE(n_components=2, **kwargs)
        tsne_results = tsne.fit_transform(data)
        real_fake = ['red' if i >= train_shape[0] else 'blue' for i in range(data.shape[0])]

        plt.figure(**self.figure_settings)
        plt.autoscale(True)

        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], color=real_fake)
        plt.savefig(self.output_path + "/" + filename)

    def pca(self, generated_data, filename):
        data, train_shape = self.merge_train_and_generated_data(generated_data)

        pca = decomposition.PCA(n_components=2)

        pca.fit(data[0:data.shape[0], :])

        X = pca.transform(data)

        matplotlib.use('Agg')

        plt.figure(**self.figure_settings)
        plt.autoscale(True)

        # plt.ylim(ymax=10, ymin=-10)
        # plt.xlim(xmax=10, xmin=-10)

        n = train_shape[0]
        real_fake = ['red' if i >= n else 'blue' for i in range(data.shape[0])]
        plt.scatter(X[:, 0], X[:, 1], color=real_fake)
        plt.savefig(self.output_path+"/"+filename)

    def load_samples(self, iters, labels=True, verboseIteration=False):
        data = False
        data_labels = False
        t = 0

        filename = self.run_path + '/' + str(iters - t).zfill(5)

        while not os.path.isfile(filename + '.npy'):
            t += 1
            if t > iters:
                raise ValueError("No log could be found.")

            filename = self.run_path + '/' + str(iters - t).zfill(5)

        print(filename)

        if type(data) == bool:
            data = np.load(filename + '.npy')
        else:
            data = np.concatenate((data, np.load(filename + '.npy')), axis=0)
        if labels and type(data_labels) != bool:
            data_labels = np.concatenate((data_labels, np.load(filename + '_labels.npy')), axis=0)
        elif labels and type(data_labels) == bool:
            data_labels = np.load(filename + '_labels.npy')

        data = self.input_data.inverse_preprocessing(data, self.config['log_transformation'],
                                                  self.input_data.scaler)

        if labels:
            return data, data_labels
        else:
            return data

    def save_pca_plots(self):

        for i in range(len(self.epochs)):
            generated_data = self.load_samples(self.epochs[i] * self.iterations_per_epoch, False)

            self.pca(generated_data, "pca_epoch"+str(self.epochs[i])+".jpg")


    def __init__(self, experiment_path, run_name, epochs=[5,10,15]):

        self.figure_settings = {
            'figsize': (6, 6)
        }
        self.epochs = epochs
        with open(experiment_path + '/config.json') as json_file:
            config = json.load(json_file)

        self.config = config
        self.run_name = run_name

        self.input_data = InputData(config['data_path'])
        
        self.input_data.preprocessing(config['log_transformation'], Scaling[config['scaling']])

        train_data, _ = self.input_data.get_raw_data()
        train_shape = train_data.shape

        self.iterations_per_epoch = int(train_shape[0] / config['mb_size'] + 1)

        self.run_path = experiment_path+"/"+run_name+"/"

        self.output_path = self.run_path+"results/"

        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)