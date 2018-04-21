import numpy as np

class IO_RDS():
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

        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        self.ro = ro
        self.pandas2ri = pandas2ri
        self.readRDS = robjects.r['readRDS']
    def get_extension(self):
        return 'rds'

class IO_NPY():
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