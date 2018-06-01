import numpy as np
import os

class IO_Main():
    def load_class_details(self, folder):
        import csv
        marker = []
        marker_names = []
        class_names = []

        with open(folder+'/class_details.csv') as csvfile:
            readCSV = csv.DictReader(csvfile, delimiter=',')
            for row in readCSV:
                marker_names.append(row['marker_gene'])
                marker.append(int(row['marker_id']))
                class_names.append(row['class'])

        return (marker, marker_names, class_names)

class IO_AUTO(IO_Main):
    def load_train_set(self, folder):
        if os.path.isfile(folder + '/train.npy'):
            return IO_NPY().load_train_set(folder)
        else:
            return IO_RDS().load_train_set(folder)

    def load_test_set(self, folder):
        if os.path.isfile(folder + '/test.npy'):
            return IO_NPY().load_test_set(folder)
        else:
            return IO_RDS().load_test_set(folder)

class IO_RDS(IO_Main):
    def load_train_set(self, folder):
        return self.load(folder+'/train.rds'),  \
               self.load(folder+'/train_labels.rds', as_matrix=True)

    def load_test_set(self, folder):
        return self.load(folder + '/test.rds'), \
               self.load(self.readRDS(folder + '/test_labels.rds'), as_matrix=True)

    def load(self, file, as_matrix=False):
        arr = self.pandas2ri.ri2py(self.readRDS(file))
        if(as_matrix):
            arr = arr.as_matrix()

        return arr

    def save(self, nparray, filename):
        if nparray.ndim == 2:
            nparray = self.ro.r.matrix(nparray, nrow=nparray.shape[0], ncol=nparray.shape[1])

        self.ro.r.assign("samples", nparray)
        self.ro.r("saveRDS(samples, file='" + filename + ".rds')")

    def __init__(self):
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri

        pandas2ri.activate()

        self.ro = ro
        self.pandas2ri = pandas2ri
        self.readRDS = ro.r['readRDS']
    def get_extension(self):
        return 'rds'

class IO_NPY(IO_Main):

    def load_train_set(self, folder):
        return self.load(folder+'/train.npy'), \
               self.load(folder+'/train_labels.npy')

    def load_test_set(self, folder):
        return self.load(folder + '/test.npy'), \
               self.load(folder + '/test_labels.npy')

    def load(self, filename):
        return np.load(filename)

    def save(self, nparray, filename):
        np.save(filename + '.npy', nparray)

    def get_extension(self):
        return 'npy'

def get_IO(name):
    if name == 'rds':
        return IO_RDS()
    elif name == 'npy':
        return IO_NPY()
    else:
        raise NotImplementedError()
