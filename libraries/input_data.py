import tensorflow as tf
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn import preprocessing
from enum import Enum

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
readRDS = robjects.r['readRDS']

class Scaling(Enum):
    minmax = 1
    standard = 2

    @staticmethod
    def get_scaler(s):
        if s == Scaling.minmax:
            return preprocessing.MinMaxScaler
        elif s == Scaling.standard:
            return preprocessing.StandardScaler
        else:
            ValueError('Wrong value')

class InputData():
    
    def get_scaler(self):
        return self.scaler
    
    def iterator(self, mb_size, i):
        np.random.seed(i*100+mb_size)
        idx = np.random.randint(0, self.train.shape[0], mb_size)
        
        return self.train[idx, :], self.train_labels[idx, :]
    
    def __scale(self, scaling=Scaling.minmax, use_raw=True):
        
        if scaling==Scaling.minmax:
            scaler = preprocessing.MinMaxScaler()
        else:
            scaler = preprocessing.StandardScaler()
        
        if use_raw:
            scaler.fit(self.train_raw)
            self.train = scaler.transform(self.train_raw)
        else:
            scaler.fit(self.train)
            self.train = scaler.transform(self.train)
            
        self.scaler = scaler
        
    def load_data(self, dataset_path, test=True):
        self.train_raw = readRDS(dataset_path+'/train.rds')
        self.train_raw = pandas2ri.ri2py(self.train_raw)
        self.train_labels = readRDS(dataset_path+'/train_labels.rds')
        self.train_labels = pandas2ri.ri2py(self.train_labels)
        
        if test:
            self.test_raw = readRDS(dataset_path+'/test.rds')
            self.test_raw = pandas2ri.ri2py(self.test_raw)
            self.test_labels = readRDS(dataset_path+'/test_labels.rds') 
            self.test_labels = pandas2ri.ri2py(self.labels)
        
    
    def __log_transform(self, use_raw=True):
        if use_raw:
            self.train = np.log2(self.train_raw + 1e-8)
        else:
            self.train = np.log2(self.train)
    
    def preprocessing(self, log_transformation=True, scaling=Scaling.minmax, use_raw=True):
        
        # Don't allow to do preprocessing twice, just to avoid possible hazards.
        
        if self.done_preprocessing:
            raise RuntimeError("Already done preprocessing")
        else:
            self.done_preprocessing = True
        
        self.log_transformation = log_transformation
        self.scaling = scaling
        
        if log_transformation:
            self.__log_transform(use_raw)
            use_raw = False

        if scaling in Scaling:
            self.__scale(scaling, use_raw)
    
    @staticmethod
    def inverse_preprocessing(data, log_transformation, scaler=None):

        if(scaler != None):
            data = scaler.inverse_transform(data)

        if(log_transformation):    
            data = np.exp2(data)
        
        return data
    
    def get_raw_data(self, train=True):
        if train:
            return self.train_raw, self.train_labels
        else:
            return self.test_raw, self.test_labels
    
    def get_data(self, train=True):
        if train:
            return self.train, self.train_labels
        else:
            return self.test, self.test_labels
    
    def __init__(self, dataset_path, test_set=False):
        self.done_preprocessing = False
        self.scaler = None
        
        self.load_data(dataset_path, test_set)
