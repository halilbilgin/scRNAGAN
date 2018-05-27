from sklearn import decomposition
from sklearn import preprocessing
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
from mpl_toolkits.mplot3d import Axes3D
from libraries.IO import IO_RDS, IO_NPY, IO_AUTO
from libraries.input_data import InputData, Scaling
#import rpy2.robjects as ro
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()
#import rpy2.robjects as robjects
#from rpy2.robjects import pandas2ri
#pandas2ri.activate()
#readRDS = robjects.r['readRDS']
import operator as o
import matplotlib.cm as cm

class Analysis(object):

    def merge_train_and_generated_data(self, epoch):

        real_data , real_labels = self.input_data.get_raw_data()
        generated_data, generated_labels = self.load_samples(epoch)

        data = np.concatenate((real_data, generated_data), axis=0)
        labels = np.argmax(np.concatenate((real_labels.astype(int), generated_labels.astype(int))), 1)
        real_fake = np.array([False if i >= real_data.shape[0] else True for i in range(data.shape[0])])

        #scaler = preprocessing.StandardScaler().fit(data)
        #data = scaler.transform(data)
        #data[0:train_data.shape[0], :] = scaler.transform(train_data)
        #data[train_data.shape[0]:, :] = scaler.transform(generated_data)
        return data, labels, real_fake

    def get_gene(self, gene, cellType, epoch):
        iteration = (epoch+1)*self.iterations_per_epoch
        generated_data, generated_labels = self.load_samples(iteration)

        train_data, labels = self.input_data.get_raw_data()
        if type(cellType) == list:
            query = np.sum(generated_labels[:, cellType], axis=1)==1
            query_train = np.sum(labels[:, cellType], axis=1) ==1
        else:
            query = generated_labels[:, cellType]==1
            query_train = labels[:, cellType]==1
        return generated_data[query, gene], \
               train_data[query_train, gene]

    def get_marker_vector(self, data, labels):

        ratio_vector = np.zeros(len(self.marker))

        for cell_type in range(len(self.marker)):
            marker = self.marker[cell_type]
            sum = labels[:, cell_type]

            ratio_vector[cell_type] = np.mean(data[np.where(sum == 0), marker]) / \
                              (np.mean(data[np.where(sum == 1), marker]))

            if np.isnan(ratio_vector[cell_type]):
                ratio_vector[cell_type] = 0
        return ratio_vector

    def get_generated_ratio(self, epoch):

        generated_data, generated_labels = self.load_samples(epoch)
        generated_ratio = self.get_marker_vector(generated_data, generated_labels)

        return generated_ratio

    def get_hyperparams(self):
        return self.config

    def get_true_ratio(self):
        train_data, train_labels = self.input_data.get_raw_data()
        true_ratio = self.get_marker_vector(train_data, train_labels)

        return true_ratio

    def get_index_scores(self, generated_ratio):

        true_ratio = self.get_true_ratio()

        return np.round(true_ratio, 3)-np.round(generated_ratio, 3)

    def plot_ratios(self, epochs):
        dpoints = []

        scores = [self.get_generated_ratio(i) for i in epochs]
        epochs.append('ORIGINAL')
        scores.append(self.get_true_ratio())

        for i in range(len(scores)):
            for j in range(len(scores[i])):
                dpoints.append([epochs[i], self.marker_names[j], scores[i][j]])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.barplot(ax, np.array(dpoints))
        plt.show()

    def barplot(self, ax, dpoints):
        '''
        Create a barchart for data across different categories with
        multiple conditions for each category.

        @param ax: The plotting axes from matplotlib.
        @param dpoints: The data set as an (n, 3) numpy array
        '''

        # Aggregate the conditions and the categories according to their
        # mean values
        conditions = [(c, np.mean(dpoints[dpoints[:, 0] == c][:, 2].astype(float)))
                      for c in np.unique(dpoints[:, 0])]
        categories = [(c, np.mean(dpoints[dpoints[:, 1] == c][:, 2].astype(float)))
                      for c in np.unique(dpoints[:, 1])]
        # sort the conditions, categories and data so that the bars in
        # the plot will be ordered by category and condition
        conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(0))]
        categories = [c[0] for c in categories]

        dpoints = np.array(sorted(dpoints, key=lambda x: categories.index(x[1])))

        # the space between each set of bars
        space = 0.3
        n = len(conditions)
        width = (1 - space) / (len(conditions))

        # Create a set of bars at each position
        for i, cond in enumerate(conditions):
            indeces = range(1, len(categories) + 1)
            vals = dpoints[dpoints[:, 0] == cond][:, 2].astype(np.float)
            pos = [j - (1 - space) / 2. + i * width for j in indeces]
            ax.bar(pos, vals, width=width, label=cond,
                   color=cm.Accent(float(i) / n))

        # Set the x-axis tick labels to be equal to the categories
        ax.set_xticks(indeces)
        ax.set_xticklabels(categories)
        plt.setp(plt.xticks()[1], rotation=90)

        # Add the axis labels
        ax.set_ylabel("Ratios")
        ax.set_xlabel("Marker Genes")

        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left')

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

    def load_samples(self, epoch, labels=True, verboseIteration=False):
        iters = (epoch+1)*self.iterations_per_epoch
        t = 0
        path = self.run_path + 'run_0/'
        filename =  str(iters - t).zfill(5)
        extension = '.' + self.IO.get_extension()
        while not os.path.isfile(os.path.join(path, filename + extension)):

            t += 1
            if t > iters:
                raise ValueError("No log could be found.")

            filename = str(iters - t).zfill(5)

        files = [os.path.join(self.run_path, 'run_0', filename)]

        i = 1
        while os.path.isfile(os.path.join(self.run_path, 'run_'+str(i), filename + extension)):
            files.append(os.path.join(self.run_path, 'run_'+str(i), filename))
            i+=1

        data = self.IO.load(files[0] + extension)
        for i in range(1, len(files)):
            data = np.concatenate((data, self.IO.load(files[i] + extension)), axis=0)

        if labels:
            data_labels = self.IO.load(files[0] + '_labels' + extension)
            for i in range(1, len(files)):
                data_labels = np.concatenate((data_labels,
                                             self.IO.load(files[i] + '_labels' + extension)),
                                             axis=0)

        #data = self.input_data.inverse_preprocessing(data, self.config['log_transformation'],
        #                                          self.input_data.scaler)

        if labels:
            return data, data_labels
        else:
            return data

    def plot_pca(self, epoch):
        data, labels, is_real = self.merge_train_and_generated_data(epoch)

        pca = decomposition.PCA(n_components=2)
        pca.fit(data[is_real, :])

        X = pca.transform(data)

        plt.scatter(X[:, 0], X[:, 1],
                    color=['green' if i else 'red' for i in is_real])

    def __init__(self, experiment_path, use_test_set=False, IO=IO_NPY()):
        self.IO = IO
        self.figure_settings = {
            'figsize': (30, 6)
        }

        with open(experiment_path + '/config.json') as json_file:
            config = json.load(json_file)

        self.config = config
        self.marker, self.marker_names, \
                    self.class_names = self.IO.load_class_details(config['data_path'])

        self.input_data = InputData(config['data_path'], IO_AUTO(), use_test_set)
        
        self.input_data.preprocessing(config['log_transformation'], None if config['scaling'] not in Scaling.__members__ else Scaling[config['scaling']])

        train_data, _ = self.input_data.get_raw_data()
        train_shape = train_data.shape

        self.iterations_per_epoch = int(train_shape[0] / config['mb_size'] + 1)

        self.run_path = experiment_path+"/"

        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(train_data)