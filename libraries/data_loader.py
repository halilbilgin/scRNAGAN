class RDSLoader():
    def load_train_set(self, folder):
        return self.pandas2ri.ri2py(self.readRDS(folder+'/train.rds')),  \
               self.pandas2ri.ri2py(self.readRDS(folder+'/train_labels.rds')).as_matrix()

    def load_test_set(self, folder):
        return self.pandas2ri.ri2py(self.readRDS(folder + '/test.rds')), \
               self.pandas2ri.ri2py(self.readRDS(folder + '/test_labels.rds')).as_matrix()

    def __init__(self):
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        self.pandas2ri = pandas2ri
        self.readRDS = robjects.r['readRDS']