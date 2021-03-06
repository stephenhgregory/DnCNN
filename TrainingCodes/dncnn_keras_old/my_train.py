# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/husqin/DnCNN-keras
# =============================================================================

# run this to train the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0, 0.9] rather than the default 0.99. 
# The Gaussian noise output helps to stablize the batch normalization, thus a small momentum is preferred.
# =============================================================================

import argparse
import re
import os, glob, datetime
import numpy as np
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import data_generator as dg
import keras.backend as K

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print(f'The following line threw an exception: tf.config.experimental.set_memory_growth(physical_devices[0], True)')
    pass

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='data/CoregisteredImages', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=300, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()


save_dir = os.path.join('models',args.model+'_'+'sigma'+str(args.sigma)) 

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def MyDnCNN(depth, filters=64, image_channels=1, use_batchnorm=True):
    """
    Complete implementation of MyDnCNN, a residual network using the Keras API.
    MyDnCNN is originally derived from DnCNN, but with some changes and
    reorganization
    :param depth: The total number of layers for the network, colloquially referred to
                    as the "depth" of the network
    :param filters: The total number of convolutional kernels in each convolutional
                    layer of the network
    :param image_channels: The number of dimensions of the input images, i.e.
                            image_channels=1 for grayscale images, or image_channels=3
                            for RGB images.
    :param use_batchnorm: Whether or not the layers of the network should use batch
                            normalization
    :return: A MyDnCNN model
    """

    # Initialize counter to keep track of current layer
    layer_index = 0

    # Define Layer 0 -- The input layer, and increment layer_index
    input_layer = Input(shape=(None, None, image_channels), name='Input'+str(layer_index))
    layer_index += 1

    # Define Layer 1 -- Convolutional Layer + ReLU activation function, and increment layer_index
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal', padding='same',
               name='Conv'+str(layer_index))(input_layer)
    layer_index += 1
    x = Activation('relu', name='ReLU'+str(layer_index))(x)

    # Iterate through the rest of the (depth - 2) layers -- Convolutional Layer + (Maybe) BatchNorm layer + ReLU
    for i in range(depth - 2):

        # Define Convolutional Layer
        layer_index += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                   use_bias=False, name='Conv' + str(layer_index))(x)

        # (Optionally) Define BatchNormalization layer
        if use_batchnorm:
            layer_index += 1
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='BatchNorm'+str(layer_index))(x)

        # Define ReLU Activation Layer
        layer_index += 1
        x = Activation('relu', name='ReLU'+str(layer_index))(x)

    # Define last layer -- Convolutional Layer and Subtraction Layer (input - noise)
    layer_index += 1
    x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               use_bias=False, name='Conv' + str(layer_index))(x)
    x = Subtract(name='Subtract'+str(layer_index))([input_layer - x])

    # Finally, define the model
    model = Model(inputs=input_layer, outputs=x)

    return model


def DnCNN(depth,filters=64,image_channels=1, use_bnorm=True):
    layer_count = 0
    input_layer = Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(input_layer)
    layer_count += 1
    x = Activation('relu',name = 'relu'+str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth-2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
        # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        x = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        layer_count += 1
        x = Activation('relu',name = 'relu'+str(layer_count))(x)  
    # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
    layer_count += 1
    x = Subtract(name = 'subtract' + str(layer_count))([input_layer, x])   # input - noise
    model = Model(inputs=input_layer, outputs=x)
    
    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.hdf5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch<=30:
        lr = initial_lr
    elif epoch<=60:
        lr = initial_lr/10
    elif epoch<=80:
        lr = initial_lr/20 
    else:
        lr = initial_lr/20 
    log('current learning rate is %2.8f' %lr)
    return lr

def train_datagen(epoch_iter=2000,epoch_num=5,batch_size=128,data_dir=args.train_data):
    while(True):
        n_count = 0
        if n_count == 0:
            # print(n_count)
            print(f'Accessing training data in: {data_dir}')
            xs = dg.datagenerator(data_dir)
            assert len(xs)%args.batch_size ==0, \
            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
            xs = xs.astype('float32')/255.0
            indices = list(range(xs.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i+batch_size]]
                noise =  np.random.normal(0, args.sigma/255.0, batch_x.shape)    # noise
                #noise =  K.random_normal(ge_batch_y.shape, mean=0, stddev=args.sigma/255.0)
                batch_y = batch_x + noise 
                yield batch_y, batch_x


def my_train_datagen(epoch_iter=2000, num_epochs=5, batch_size=128, data_dir=args.train_data):
    """
    Generator function that yields training data samples from a specified data directory

    :param epoch_iter: The number of iterations per epoch
    :param num_epochs: The total number of epochs
    :param batch_size: The number of training examples for each training iteration
    :param data_dir: The directory in which training examples are stored
    :return: Yields a training example x and noisey image y
    """




        
# define loss
def sum_squared_error(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return K.sum(K.square(y_pred - y_true))/2
    
if __name__ == '__main__':
    # model selection
    model = DnCNN(depth=17,filters=64,image_channels=1,use_bnorm=True)
    model.summary()
    
    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:  
        print('resuming by loading epoch %03d'%initial_epoch)
        model = load_model(os.path.join(save_dir,'model_%03d.hdf5'%initial_epoch), compile=False)
    
    # compile the model
    model.compile(optimizer=Adam(0.001), loss=sum_squared_error)
    
    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'model_{epoch:03d}.hdf5'), 
                verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(save_dir,'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    history = model.fit_generator(train_datagen(batch_size=args.batch_size),
                steps_per_epoch=2000, epochs=args.epoch, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler])

















