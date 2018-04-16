import tensorflow as tf
import numpy as np
from libraries.input_data import Scaling
from libraries.acgan import ACGAN
from libraries.input_data import InputData
from libraries.IO import get_IO

def sample_z(m, n):
    return np.random.normal(0,1, size=[m, n])

def cross_entropy(logit, y):
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))

def check_scaling(s):
    if s in Scaling.__members__.keys():
        return s
    else:
        raise NotImplementedError("No such scaling method implemented")

def get_acgan(config):
    if 'leaky_param' not in config:
        config['leaky_param'] = 0.1

    config['activation_function'] = get_activation(config['activation_function'], config['leaky_param'])
    config['generator_output_activation'] = get_activation('tanh' if config['scaling'] == 'minmax' else 'none')

    if 'wgan' not in config:
        config['wgan'] = False

    if 'normalizer_fn' not in config :
        config['normalizer_fn'] = None
    else:
        config['normalizer_fn'] = tf.contrib.layers.batch_norm
        config['normalizer_params'] = {'center': True, 'scale': True}

    if 'IO' not in config:
        config['IO'] = 'npy'

    IO = get_IO(config['IO'])

    input_data = InputData(config['data_path'], IO)

    if config['scaling'] not in Scaling.__members__:
        scaling = None
    else:
        scaling = Scaling[config['scaling']]
    input_data.preprocessing(config['log_transformation'], scaling)

    train_data, train_labels = input_data.get_data()

    return ACGAN(train_data.shape[1], train_labels.shape[1], input_data, **config)

def check_activation_function(s):
    if(s in ['sigmoid', 'leaky_relu', 'relu', 'none']):
        return s
    else:
        raise NotImplementedError("No such activation function exists")

class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

def get_activation(s, alpha=0.1):
    if(s == 'leaky_relu'):
        return lambda x : tf.maximum(x, alpha * x)
    elif(s == 'relu'):
        return tf.nn.relu
    elif(s == 'tanh'):
        return tf.nn.tanh
    elif(s == 'sigmoid'):
        return tf.nn.sigmoid
    elif(s == 'none'):
        return None
    else:
        raise NotImplementedError("No such activation is implemented")

def leaky_relu(x, alpha=0.1):
    return tf.maximum(x, alpha * x)