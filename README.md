# scRNAGAN

### Creating and running an experiment using script.py:

Specifify the hyperparameters and the location of dataset in script.py

Example:

    config = {

        "data_path": ["/home/halilbilgin/data/alphabeta_joint_500/"], # the location of train.npy and train_labels.npy
        "log_transformation": [0],                                     # whether log transformation will be done before training or not
        "scaling": ["minmax"],                                         # minmax, standard (z-score) or none scaling before training
        "d_hidden_layers": [[1500, 200], [120, 450]],                  # discriminator's hidden layer architecture
        "g_hidden_layers": [[1000], [250, 500]],                       # generator's hidden layer architecture
        "activation_function": ["tanh"],                               # activation function will be used in hidden layers 
        "leaky_param": [0.1],                                          # alpha parameter if leaky ReLU is used
        "learning_rate": [0.00001, 0.0001, 0.001, 0.01],               # learning rate that will be used
        "learning_schedule": ['no_schedule', 'search_then_converge'],  # learning schedule
        "optimizer": [ 'Adam', 'Adadelta'],                            # optimizer
        "wgan": [1],                                                   # whether the loss function is classical GAN or WGAN-GP
        "z_dim": [100],                                                # dimension of input noise given to generator 
        "mb_size": [40, 60, 80, 100],                                  # minibatch size
        "d_dropout": [0],                                              # dropout applied to discriminator's hidden layers
        "g_dropout": [0],                                              # dropout applied to generator's hidden layers
        "label_noise": [0]                                             # label noise to (see https://github.com/soumith/ganhacks )

    }
    ..
    ..
    ..
    args.epochs = 30                                                   # total number of epochs
    args.log_sample_freq = 5                                           # how frequent you want to log the samples? (e.g 5 epoch)
    args.log_sample_size = 500                                         # how many samples should be generated from generator (e.g 500) 
    .
    .
    .
    repeat = 3                                                         # how many times do you want to repeat the same experiment?

Then, run
    
    python script.py --exp_dir /vol1/ibrahim/out/gene_500_batchsize
    or
    sbatch script.py --exp_dir /vol1/ibrahim/out/gene_500_batchsize #if you use slurm
where --exp_dir specifies where the experiment should be located

When experiment run completely, you can execute:
    
    python analysis.py --exp_dir /vol1/ibrahim/out/gene_500_batchsize

to generate PCA plots, marker plots and index scores. All the results will be stored in /vol1/ibrahim/out/gene_500_batchsize/analysis/ 

*config* allows grid search. For example if you set `"mb_size": [40, 60, 80, 100]` then all the possible minibatch sizes will run in a separate model. In the grid search configuration above, *script.py* creates 256 different hyperparameter combinations (i.e. 2 possible *d_hidden_layers*, *g_hidden_layers*, *learning_schedule*, *optimizer* and 4 possible *learning_rate* & *mb_size*, 2*2*2*2*4*4 = 256) 

All the hyperparamter combinations created using grid search will be stored in a subfolder named with a random generated ID. For example, if the *--exp_dir* is */vol1/ibrahim/out/gene_500_batchsize* then, a possible hyperparameter combination could be stored in the folder "/vol1/ibrahim/out/gene_500_batchsize/h_bn_BDBSUMVLWQ". 

The ID's and hyperparameters are stored in config.json in the subfolder
In addition, analysis.py creates a results.csv where the ID, hyperparameters of the experiment with the ID and index stores are stored. 

### Dataset

1. train.npy
2. train_labels.npy
3. class_details.csv

train.npy -> The training dataset you will use should be saved either as npy or rds file. If it is rds, you should change *args.IO* to *'rds'* in script.py Rows should be samples, columns should represent features

train_labels.npy -> allows to use classes both in training and analysis. Columns represent labels in one-hot vector format.

class_details.csv-> columns represent: class name, marker gene and marker id (column number in train.npy)

A dataset I used in the presentation is located in data/alpha_beta_joint_500

### Reproducing the experiments shown in the slides 

Below is command line way of creating and running an experiment of which hyperparameter configuration file is saved as a JSON file. 

You can access the files used in my presentation from *out* folder of this repo.

In the out/slides_3layers/exp.json, replace the data path with the absolute data path in your computer and use that absolute path in the commands as well. 

    python create_experiments -cfg /home/.../scRNAGAN/out/slides_3layers/exp.json -epath /home/.../scRNAGAN/out/slides_3layers/
    python train_all.py -epath /home/.../scRNAGAN/out/slides_3layers/ -repeat 4 -epochs 30 -l_freq 5 -l_size 500
    python analysis.py -exp_dir /home/.../scRNAGAN/out/slides_3layers/

### Running differential gene expression using generated samples after training the model

    from libraries.analysis import Analysis
    analysis = Analysis('/home/.../scRNAGAN/out/slides_3layers/h_bn_BDBSUMVLWQ')
    
    results = analysis.differential_gene_expression(20)
    
results variable stores the percentage of differentially expressed genes for each cell type of generated samples in 20th epoch.

### Prerequisites

What things you need to install the software and how to install them

Matplotlib

Numpy

Tensorflow >= 1.3

Scikit-learn

Enum34 (for python 2.7)

rpy2(optional, for differential gene expression analysis)
