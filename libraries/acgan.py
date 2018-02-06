import tensorflow as tf
import numpy as np
import datetime
import os
from tensorflow.contrib.layers import fully_connected
from libraries.utils import sample_z, cross_entropy, objdict
import sys
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

class ACGAN():
    
    def discriminator(self, X):
        config = self.config
        reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
        
        connected = X
        
        for i in range(len(config.d_hidden_layers)):
            connected = tf.layers.dropout(fully_connected(connected, config.d_hidden_layers[i],
                        config.activation_function,
                        weights_regularizer = config.weights_regularizer,
                        reuse=tf.AUTO_REUSE, scope='discriminator_'+str(i)), config.d_dropout, training=self.phase)
        if config.wgan:
            out_gan = fully_connected(connected, num_outputs=1,
                                  activation_fn=None
                                  , reuse=tf.AUTO_REUSE, scope='discriminator_out_gan')
        else:
            out_gan = fully_connected(connected, num_outputs=1,
                                      activation_fn=tf.nn.sigmoid
                                      , reuse=tf.AUTO_REUSE, scope='discriminator_out_gan')

        out_aux = fully_connected(connected, num_outputs=config.y_dim, activation_fn=None
                                    , reuse=tf.AUTO_REUSE, scope='discriminator_out_aux')
        
        return out_gan, out_aux

    def Batchnorm(self, name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True,
                  labels=None, n_labels=None):
        """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
        if axes != [0, 2, 3]:
            raise Exception('unsupported')
        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        shape = mean.get_shape().as_list()  # shape is [1,n,1,1]
        offset_m = tf.Variable(name + '.offset', np.zeros([n_labels, shape[1]], dtype='float32'))
        scale_m = tf.Variable(name + '.scale', np.ones([n_labels, shape[1]], dtype='float32'))
        offset = tf.nn.embedding_lookup(offset_m, labels)
        scale = tf.nn.embedding_lookup(scale_m, labels)
        result = tf.nn.batch_normalization(inputs, mean, var, offset[:, :, None, None], scale[:, :, None, None], 1e-5)
        return result

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
        if not config.closs_scale_g:
            config.closs_scale_g = 0.1

        GC_loss = G_loss - config.closs_scale_g*C_loss

        return D_loss, G_loss, DC_loss, GC_loss, C_loss
        
    def get_optimizers(self, DC_loss, GC_loss, return_grads = True):
        config = self.config
        theta_D = [t for t in tf.trainable_variables() if t.name.startswith('discriminator')]        
        theta_G = [t for t in tf.trainable_variables() if t.name.startswith('generator')]

        if config.wgan:
            decay = tf.maximum(0., 1. - (tf.cast(self._iteration, tf.float32) / self.totalIteration))

            opt = tf.train.AdamOptimizer(learning_rate=config.lr*decay, beta1=0, beta2=0.9)
        else:
            opt = tf.train.AdamOptimizer(learning_rate=config.lr)
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
    
    def close_session(self):
        tf.reset_default_graph()
        self.sess.close()

    def get_config(self):
        return self.config
    
    def train_and_log(self, next_batch, logs_path, iterations = 3000, summary_freq=10, print_freq=20,
                      log_sample_freq=150, log_sample_size=200, d_steps=1, sample_z=sample_z):
        config = self.config
        sess = tf.Session()
        self.sess = sess

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logs_path,
                                                    graph=tf.get_default_graph())

        for it in range(iterations):

            for d_step in range(d_steps):
                X_mb, y_mb = next_batch(config.mb_size, (d_step+1)*it)
                z_mb = sample_z(config.mb_size, config.z_dim)

                _, DC_loss_curr, c_acc, f_acc = sess.run(
                    [self.D_solver, self.DC_loss, self.discriminator_class_accuracy, self.discriminator_fake_accuracy],

                    feed_dict={self.X: X_mb, self.y: y_mb, self.z: z_mb, self.phase: 1,
                               self._iteration: it, self.totalIteration: iterations}
                )

            _, GC_loss_curr, summary = sess.run(
                [self.G_solver, self.GC_loss, merged_summary_op],
                feed_dict={self.X: X_mb, self.y: y_mb, self.z: z_mb, self.phase: 1,
                               self._iteration: it, self.totalIteration: iterations}
            )

            if it % summary_freq == 0:
                summary_writer.add_summary(summary, it)

            if it % print_freq == 0:
                print('Iter: {}; DC_loss: {:.4}; GC_loss: {:.4}; Discfakeacc: {:.2}; Discclassacc: {:.2};'
                      .format(it, DC_loss_curr, GC_loss_curr, f_acc, c_acc ))

            if(DC_loss_curr > 1000):
                return False

            if it % log_sample_freq == 0:
                idx = np.random.randint(0, config.y_dim, log_sample_size)
                c = np.zeros([log_sample_size, config.y_dim])
                c[range(log_sample_size), idx] = 1

                samples = sess.run(self.G_sample, feed_dict={self.z: sample_z(log_sample_size, config.z_dim),
                                                     self.y: c, self.phase: 0})
                samples = self.input_data.inverse_preprocessing(samples, config.log_transformation, self.input_data.scaler)

                samples = ro.r.matrix(samples, nrow=samples.shape[0], ncol=samples.shape[1])

                ro.r.assign("samples", samples)
                ro.r.assign("c", c)
                filename = logs_path+'/'+'{}'.format(str(it).zfill(5))
                ro.r("saveRDS(samples, file='"+filename+".rds')")
                ro.r("saveRDS(c, file='" + filename+ "_labels.rds')")

    def __init__(self, X_dim, y_dim, input_data, **kwargs):
        self.input_data = input_data
        default_config = {
            'd_hidden_layers': [180, 45],
            'g_hidden_layers': [50, 200],
            'normalizer_fn': None,
            'weights_regularizer': None,
            'activation_function': tf.nn.leaky_relu,
            'g_dropout': 0.3,
            'd_dropout': 0.5,
            'label_noise': 0.3,
            'mb_size': 10,
            'eps': 1e-8,
            'lr':4e-4,
            'normalizer_params':{},
            'X_dim': X_dim,
            'y_dim': y_dim,
            'z_dim': kwargs['z_dim'],
            'generator_output_activation': kwargs['generator_output_activation']
        }

        self.config = {**default_config, **kwargs}
        self.config = objdict(self.config)
