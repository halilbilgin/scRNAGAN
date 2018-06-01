import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.layers import fully_connected
from libraries.utils import sample_z, cross_entropy, objdict, get_activation, get_optimizer, get_learning_schedule
from libraries.input_data import InputData, Scaling
from libraries.IO import get_IO
import json

class ACGAN():
    
    def discriminator(self, X):
        config = self.config

        connected = X
        
        for i in range(len(config.d_hidden_layers)):
            connected = tf.layers.dropout(fully_connected(connected, config.d_hidden_layers[i],
                        config.activation_function,
                        weights_regularizer = config.weights_regularizer,
                        reuse=tf.AUTO_REUSE, scope='discriminator_'+str(i)), float(config.d_dropout), training=self.phase)

        out_gan = fully_connected(connected, num_outputs=1,
                                  activation_fn=None if config.wgan else tf.nn.sigmoid
                                  , reuse=tf.AUTO_REUSE, scope='discriminator_out_gan')

        out_aux = fully_connected(connected, num_outputs=config.y_dim, activation_fn=None
                                    , reuse=tf.AUTO_REUSE, scope='discriminator_out_aux')
        
        return out_gan, out_aux

    def generator(self, z, c):
        
        config = self.config

        connected = tf.concat(axis=1, values=[z, c])

        for i in range(len(config.g_hidden_layers)):
            connected = tf.layers.dropout(fully_connected(connected, config.g_hidden_layers[i],
                            config.activation_function, config.normalizer_fn,
                            normalizer_params=config.normalizer_params,
                            weights_regularizer = config.weights_regularizer,
                            reuse=tf.AUTO_REUSE, scope='generator_'+str(i)), config.g_dropout, training=self.phase)
        
        return fully_connected(connected, config.X_dim, 
                               activation_fn=config.generator_output_activation, 
                               reuse=tf.AUTO_REUSE, scope='generator_out')

    def get_losses(self, D_real, C_real, D_fake, C_fake):
        config = self.config
        C_loss = cross_entropy(C_real, self.y) + cross_entropy(C_fake, self.y)

        if config.wgan:
            G_loss = -tf.reduce_mean(D_fake)
            D_loss = tf.reduce_mean(D_fake  - D_real)

            epsilon = tf.random_uniform(
                shape=[config.mb_size, 1],
                minval=0.,
                maxval=1.
            )
            X_hat = self.X + epsilon * (self.G_sample - self.X)

            D_X_hat = self.discriminator(X_hat)
            grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]

            slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=[1]))

            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            D_loss += 10 * gradient_penalty

        else:
            D_loss = -tf.reduce_mean(tf.log(tf.maximum(D_real + tf.random_uniform([config.mb_size,1],
                                                  -config.label_noise,config.label_noise), 0) + config.eps) +
                                    tf.log(tf.maximum(1. - D_fake + tf.random_uniform([config.mb_size,1],
                                                  -config.label_noise,config.label_noise), 0) + config.eps))

            G_loss = -tf.reduce_mean(tf.log(D_fake + config.eps))

        DC_loss = D_loss - C_loss
        if 'closs_scale_g' not in config or not config.closs_scale_g:
            config.closs_scale_g = 0.1

        GC_loss = G_loss - config.closs_scale_g*C_loss

        return D_loss, G_loss, DC_loss, GC_loss, C_loss
        
    def get_optimizers(self, DC_loss, GC_loss, return_grads = True):
        config = self.config
        theta_D = [t for t in tf.trainable_variables() if t.name.startswith('discriminator')]        
        theta_G = [t for t in tf.trainable_variables() if t.name.startswith('generator')]

        if config.wgan:
            decay = tf.maximum(0., 1. - (tf.cast(self._iteration, tf.float32) / self.totalIteration))

            if not isinstance(config['optimizer'], tf.train.AdamOptimizer):
                opt = config['optimizer'](learning_rate=config.lr*decay)
            else:
                opt = config['optimizer'](learning_rate=config.lr*decay, beta1=0, beta2=0.9)

        else:
            opt = config['optimizer'](learning_rate=config.learning_schedule(config.lr, self._iteration))

        with tf.variable_scope('optimizers', reuse=tf.AUTO_REUSE):

            D_grads = opt.compute_gradients(DC_loss, theta_D,
                            colocate_gradients_with_ops= True if config.wgan else False)
            D_solver = opt.apply_gradients(D_grads)

            G_grads = opt.compute_gradients(GC_loss, var_list=theta_G,
                            colocate_gradients_with_ops=True if config.wgan else False)
            G_solver = opt.apply_gradients(G_grads)

        if(return_grads):
            return D_solver, G_solver, D_grads, G_grads
        else:
            return D_solver, G_solver
    
    def load_summary(self):
        
        for grad, var in self.D_grads + self.G_grads:
            tf.summary.histogram(var.name.replace(':','_') + '/gradient', grad)
        
        self.discriminator_fake_accuracy = tf.reduce_mean(
                tf.cast(tf.concat([tf.greater_equal(self.D_real,0.5), tf.less(self.D_fake,0.5)], 0), 
                        tf.float32))
        
        self.discriminator_class_accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(
                        tf.concat([tf.nn.softmax(self.C_real), tf.nn.softmax(self.C_fake)], 0), axis = 1),
                    tf.argmax(
                        tf.concat([self.y, self.y], 0), axis = 1)),
                tf.float32))

        tf.summary.scalar("Discriminatorfake_accuracy", self.discriminator_fake_accuracy)
        tf.summary.scalar("Discriminatorclass_accuracy", self.discriminator_class_accuracy)

        tf.summary.scalar("D_loss", self.D_loss)
        tf.summary.scalar("G_loss", self.G_loss)
        tf.summary.scalar("DC_loss", self.DC_loss)
        tf.summary.scalar("GC_loss", self.GC_loss)
        tf.summary.scalar("C_loss", self.C_loss)

    def build_model(self):
        config = self.config
        with tf.variable_scope("placeholders", reuse=tf.AUTO_REUSE):
            self.X = tf.placeholder(tf.float32, shape=[None, config.X_dim], name='InputData')
            self.y = tf.placeholder(tf.float32, shape=[None, config.y_dim], name='LabelData')
            self.z = tf.placeholder(tf.float32, shape=[None, config.z_dim], name='GeneratorPriors')
            self.phase = tf.placeholder(tf.bool, name='phase')
            self._iteration = tf.placeholder(tf.int32, shape=None)
            self.totalIteration = tf.placeholder(tf.float32, shape=None)
        config.normalizer_params['is_training'] = self.phase
        
        self.G_sample = self.generator(self.z, self.y)
        
        self.D_real, self.C_real = self.discriminator(self.X)
        self.D_fake, self.C_fake = self.discriminator(self.G_sample)
        
        self.D_loss, self.G_loss, self.DC_loss, self.GC_loss, self.C_loss = self.get_losses(self.D_real,
                                                                        self.C_real, self.D_fake, self.C_fake)
        
        self.D_solver, self.G_solver, self.D_grads, self.G_grads = self.get_optimizers(self.DC_loss, 
                                            self.GC_loss, return_grads = True)       
        
        self.load_summary()

    def get_config(self):
        return self.config

    @staticmethod
    def load(sess, config):
        default_config = {
            'd_steps' : 1,
            'd_hidden_layers': [180, 45],
            'g_hidden_layers': [50, 200],
            'normalizer_fn': None,
            'weights_regularizer': None,
            'activation_function': tf.nn.relu,
            'g_dropout': 0.3,
            'd_dropout': 0.5,
            'seed': 23,
            'label_noise': 0.3,
            'leaky_param': 0.1,
            'mb_size': 10,
            'eps': 1e-8,
            'lr':4e-4,
            'normalizer_params':{},
            'wgan': False,
            'IO': 'npy',
            'optimizer': 'Adam',
            'learning_schedule': 'no_schedule',
            'log_transformation': 0
        }


        default_config.update(config)

        config = default_config

        IO = get_IO(config['IO'])

        input_data = InputData(config['data_path'], IO)

        if config['scaling'] not in Scaling.__members__:
            scaling = None
        else:
            scaling = Scaling[config['scaling']]

        config['optimizer'] = get_optimizer(config['optimizer'])

        config['learning_schedule'] = get_learning_schedule(config['learning_schedule'])

        input_data.preprocessing(config['log_transformation'], scaling)

        train_data, train_labels = input_data.get_data()

        acgan = ACGAN(sess, train_data.shape[1], train_labels.shape[1], **config)
        acgan.build_model()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        return acgan, input_data

    @staticmethod
    def load_saved_model(sess, path):
        with open(path + '/config.json') as json_file:
            config = json.load(json_file)

        config['experiment_path'] = path
        acgan = ACGAN.load(config, sess)
        acgan.build_model()
        # load the saved model
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')

        return acgan

    def train_and_log(self, logs_path, IO, input_data, iterations=3000, summary_freq=10, print_freq=20,
                      log_sample_freq=150, log_sample_size=200, sample_z=sample_z):
        sess = self.sess
        config = self.config

        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logs_path,
                                               graph=tf.get_default_graph())

        for it in range(iterations):

            for d_step in range(config.d_steps):
                curIteration = (d_step + 1) * it
                X_mb, y_mb = input_data.iterator(config.mb_size, curIteration, seed=config.seed)
                z_mb = sample_z(config.mb_size, config.z_dim)

                _, DC_loss_curr, aux_acc, gan_acc = sess.run(
                    [self.D_solver, self.DC_loss, self.discriminator_class_accuracy,
                     self.discriminator_fake_accuracy],

                    feed_dict={self.X: X_mb, self.y: y_mb, self.z: z_mb, self.phase: True,
                               self._iteration: it, self.totalIteration: iterations}
                )

            _, GC_loss_curr, summary = sess.run(
                [self.G_solver, self.GC_loss, merged_summary_op],
                feed_dict={self.X: X_mb, self.y: y_mb, self.z: z_mb, self.phase: True,
                           self._iteration: it, self.totalIteration: iterations}
            )

            if it % summary_freq == 0:
                summary_writer.add_summary(summary, it)

            if it % print_freq == 0:
                print('Iter: {}; DC_loss: {:.4}; GC_loss: {:.4}; Discfakeacc: {:.2}; Discclassacc: {:.2};'
                      .format(it, DC_loss_curr, GC_loss_curr, gan_acc, aux_acc))

            if (DC_loss_curr > 1000):
                break

            if it % log_sample_freq == 0 or it == iterations-1:
                samples, c = self.generate_samples(log_sample_size)
                samples = input_data.inverse_preprocessing(samples)

                filename = logs_path + '/' + '{}'.format(str(it).zfill(5))

                IO.save(samples, filename)
                IO.save(c, filename + '_labels')

    def save_session(self, path):
        # save the session
        saver = tf.train.Saver()
        saver.save(self.sess, path + '/model.ckpt')

    def generate_samples(self, n_size):

        y_dim = self.config.y_dim
        z_dim = self.config.z_dim

        idx = np.random.randint(0, y_dim, n_size)
        c_samples = np.zeros([n_size, y_dim])
        c_samples[range(n_size), idx] = 1

        X_samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(n_size, z_dim),
                                                            self.y: c_samples, self.phase: False})

        return X_samples, c_samples

    def __init__(self, sess, X_dim, y_dim, **kwargs):
        self.sess = sess

        config = kwargs
        config['X_dim'] = X_dim
        config['y_dim'] = y_dim

        config['activation_function'] = get_activation(config['activation_function'], config['leaky_param'])
        config['generator_output_activation'] = get_activation('tanh' if config['scaling'] == 'minmax' else 'none')

        if config['normalizer_fn']:
            config['normalizer_fn'] = tf.contrib.layers.batch_norm
            config['normalizer_params'] = {'center': True, 'scale': True}

        self.config = objdict(config)

