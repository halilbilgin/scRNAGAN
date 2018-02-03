from sklearn import decomposition
from sklearn import preprocessing
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
from mpl_toolkits.mplot3d import Axes3D

from libraries.input_data import InputData, Scaling
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
readRDS = robjects.r['readRDS']


class Analysis(object):

    def merge_train_and_generated_data(self, iteration):
        real_data , real_labels = self.input_data.get_raw_data()
        generated_data, generated_labels = self.load_samples(iteration)

        data = np.concatenate((real_data, generated_data), axis=0)
        labels = np.argmax(np.concatenate((real_labels.astype(int), generated_labels.astype(int))), 1)
        real_fake = np.array([False if i >= real_data.shape[0] else True for i in range(data.shape[0])])

        #scaler = preprocessing.StandardScaler().fit(data)
        #data = scaler.transform(data)
        #data[0:train_data.shape[0], :] = scaler.transform(train_data)
        #data[train_data.shape[0]:, :] = scaler.transform(generated_data)
        return data, labels, real_fake

    def get_gene(self, gene, cellType, iteration):
        generated_data, generated_labels = self.load_samples(iteration)

        train_data, labels = self.input_data.get_raw_data()

        return generated_data[generated_labels[:, cellType]==1, gene], \
               train_data[labels[:, cellType]==1, gene]

    def get_marker_vector(self, data, labels):

        ratio_vector = np.zeros(labels.shape[1])

        for i in range(labels.shape[1]):
            ratio_vector[i] = np.median(data[labels[:, i] == 0, self.marker[i%7]]) / \
                              (np.median(data[labels[:, i] == 1, self.marker[i%7]]))
            if np.isnan(ratio_vector[i]):
                ratio_vector[i] = 0
        return ratio_vector

    def print_ratios(self, gene, iterations):
        for i in iterations:
            generated_data, generated_labels = self.load_samples(i)

            train_data, train_labels = self.input_data.get_raw_data()

            test_ratio = self.get_marker_vector(train_data, train_labels)

            generated_ratio = self.get_marker_vector(generated_data, generated_labels)

            print(np.round(np.linalg.norm(generated_ratio-test_ratio), 3), end=', ')



    def euclidean_distance(self, epoch, normalize=True):
        generated_data, generated_labels = self.load_samples(epoch * self.iterations_per_epoch)

        train_data, labels = self.input_data.get_raw_data()

        if normalize== True:
            train_data = self.scaler.transform(train_data)
            generated_data = self.scaler.transform(generated_data)

        if epoch == 0:
            np.random.seed(2)
            ind = np.random.choice(train_data.shape[0], (generated_data.shape[0],))
            generated_data = train_data[ind, :]
            train_data = np.delete(train_data, ind, axis=0)

        np.random.seed(1)
        ind = np.random.choice(train_data.shape[0], (generated_data.shape[0],))

        distances = np.linalg.norm(train_data[ind, :] - generated_data, axis=1)
        return np.mean(distances)

    def pca(self, data, labels, is_real):

        pca = decomposition.PCA(n_components=2)
        pca.fit(data[is_real, :])

        X = pca.transform(data)

        return X

        #plt.savefig(self.output_path+"/"+filename)

    def load_samples(self, iters, labels=True, verboseIteration=False):
        data = False
        data_labels = False
        t = 0

        filename = self.run_path + '/' + str(iters - t).zfill(5)

        while not os.path.isfile(filename + '.rds'):
            t += 1
            if t > iters:
                raise ValueError("No log could be found.")

            filename = self.run_path + '/' + str(iters - t).zfill(5)


        if type(data) == bool:
            data = pandas2ri.ri2py(readRDS(filename + '.rds'))
        else:
            data = np.concatenate((data, pandas2ri.ri2py(readRDS(filename + '.rds'))), axis=0)
        if labels and type(data_labels) != bool:
            data_labels = np.concatenate((data_labels, pandas2ri.ri2py(readRDS((filename + '_labels.npy')))), axis=0)
        elif labels and type(data_labels) == bool:
            data_labels = pandas2ri.ri2py(readRDS((filename + '_labels.rds')))

        #data = self.input_data.inverse_preprocessing(data, self.config['log_transformation'],
        #                                          self.input_data.scaler)

        if labels:
            return data, data_labels
        else:
            return data

    def normal(self, iteration):
        data, labels, is_real = self.merge_train_and_generated_data(iteration)
        indices = labels < 7

        X = self.pca(data[indices], labels[indices], is_real[indices])

        plt.scatter(X[:, 0], X[:, 1],
                    color=['green' if i else 'red' for i in is_real[indices]])

    def diabetic(self, iteration):
        data, labels, is_real = self.merge_train_and_generated_data(iteration)
        indices = labels >= 7

        X = self.pca(data[indices], labels[indices], is_real[indices])

        plt.scatter(X[:, 0], X[:, 1],
                    color=['green' if i else 'red' for i in is_real[indices]])

    def cell_types(self, iteration):
        data, labels, is_real = self.merge_train_and_generated_data(iteration)
        markers = ['o', 's', '+', '.', '<', '>', "*"]
        X = self.pca(data, labels, is_real)
        for i in np.unique(labels):
            indices = np.where(labels == i)[0]
            plt.scatter(X[indices, 0], X[indices, 1],
                        color=['red' if i else 'blue' for i in is_real[indices]],
                        marker=markers[i % 7])

    def save_pca_plots(self, iterations, plotter):
        plt.figure(**self.figure_settings)
        for i in range(len(iterations)):
            plt.subplot(1,len(iterations), i+1)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            plotter(iterations[i])
            # plt.figure(**self.figure_settings)


        plt.show()

    def __init__(self, experiment_path, run_name, use_test_set=False):

        self.figure_settings = {
            'figsize': (30, 6)
        }
        self.marker = [4, 1, 0, 2, 5, 3, 6]

        with open(experiment_path + '/config.json') as json_file:
            config = json.load(json_file)

        self.config = config
        self.run_name = run_name

        self.input_data = InputData(config['data_path'], use_test_set)
        
        self.input_data.preprocessing(config['log_transformation'], None if config['scaling'] not in Scaling.__members__ else Scaling[config['scaling']])

        train_data, _ = self.input_data.get_raw_data()
        train_shape = train_data.shape

        self.iterations_per_epoch = int(train_shape[0] / config['mb_size'] + 1)

        self.run_path = experiment_path+"/"+run_name+"/"

        self.output_path = self.run_path+"results/"

        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(train_data)

        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)