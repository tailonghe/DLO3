"""
@author: Tai-Long He (tailong.he@mail.utoronto.ca)
"""

import os
import sys
import random
import warnings
import glob
import netCDF4 as nc

import numpy as np
import pandas as pd

from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, Permute, Reshape, LeakyReLU
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import normalize
from keras.regularizers import l2
import keras.layers as kl

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.metrics import mean_squared_error
import tensorflow as tf
from keras.callbacks import CSVLogger

from numpy.random import seed


def r2_keras(y_true, y_pred):
    """ Function to compute r2 correlation at non-zero locations
    """
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))

    SS_res =  K.sum(K.square(y_t - y_p)) 
    SS_tot = K.sum(K.square(y_t - K.mean(y_t))) 

    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
  

def mymse(y_true, y_pred):
    """ Function to calculate mean squared error loss with non-zero values only
    """
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))
  
    return K.sum(K.square(y_p - y_t), axis=-1)



# 1980-2005
Y_train = np.load('Y_train.npy')
# 2005-2016
Y_test = np.load('Y_test.npy')

# 1980-2009
X_train = np.load('X_train_final_1980-2009.npy')
# 2005-2014
X_test = np.load('X_test_final_2005-2014.npy')


print("X_train: ", X_train.shape, np.max(X_train[:, :, :, 0]), np.min(X_train[:, :, :, 0]))
print("Y_train: ", Y_train.shape, np.max(Y_train[:, :, :, 0]), np.min(Y_train[:, :, :, 0]))


def new_model(seed_now, path):
    seed(seed_now)
    print("Seed now: ", seed_now)

    # Input layer: the input data has shape: (None, 48, 120, 13),
    # corresponding to (batch_size, height, width, predictors)
    inputs = Input( ( 48, 120, 13 ), name='model_input')

    s = Lambda(lambda x: x / 1) (inputs) # no scaling for now

    c1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='Block1_Conv1') (s)    # 48 x 120
    c1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block1_Conv2') (c1)   # 48 x 120
    p1 = MaxPooling2D((2, 2), name='Block1_MaxPool') (c1)   # 24 x 60

    c2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block2_Conv1') (p1)   # 24 x 60
    c2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='Block2_Conv2') (c2)   # 24 x 60
    p2 = MaxPooling2D((2, 2), name='Block2_MaxPool') (c2)   # 12 x 30

    c3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='Block3_Conv1') (p2)   # 12 x 30
    c3 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='Block3_Conv2') (c3)   # 12 x 30
    p3 = MaxPooling2D((2, 2), name='Block3_MaxPool') (c3)  # 6 x 15

    c4 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='Block4_Conv1') (p3) # 6 x 15
    c4 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='Block4_Conv2') (c4) # 6 x 15

    # flatten tensor to 1d
    c4 = Permute((3, 1, 2), name='Block4_Permute1') (c4)
    c4 = Reshape((-1, 90), name='Block4_Reshape') (c4)
    f4 = Permute((2, 1), name='Block4_Permute2') (c4)  

    # 3 stacked LSTM cells
    lstm = LSTM(1024, return_sequences=True, name='LSTM1') (f4)
    lstm = LSTM(1024, return_sequences=True, name='LSTM2') (lstm)
    lstm = LSTM(1024, return_sequences=True, name='LSTM3') (lstm)

    # reshape back to 3d tensor
    resh = kl.Reshape( (6 , 15, 1024) , name='Block5_Reshape') (lstm)

    # Up-Convolutional
    u5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', name='Block5_UpConv') (resh)  # 256 x 12 x 30
    # crop some info out from the input half
    c2_cropped = Lambda(lambda x: tf.slice(x, [0, 6, 15, 0], [-1, 12, 30, -1]))(c2)    
    # residual connection     
    u5 = concatenate([u5, c3, c2_cropped])  # 512 x 12 x 30
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block5_Conv1') (u5)  # 256 x 12 x 30
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block5_Conv2') (c5)  # 256 x 12 x 30

    # Up-Convolutional
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='Block6_UpConv') (c5)  # 128 x 60 x 24
    # crop some info out from the input half
    c1_cropped = Lambda(lambda x: tf.slice(x, [0, 12, 30, 0], [-1, 24, 60, -1]))(c1)
    # residual connection 
    u6 = concatenate([u6, c2, c1_cropped])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='Block6_Conv1') (u6)  # 60 x 24
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='Block6_Conv2') (c6)  # 60 x 24

    # Up-Convolutional
    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='Block7_UpConv') (c6)  # 128 x 120 x 48
    # residual connection 
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', name='Block7_Conv1') (u7)  # 120 x 48
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', name='Block7_Conv2') (c7)  # 120 x 48

    # Output layer: the input data has shape: (None, 48, 120, 1),
    # corresponding to (batch_size, height, width, predictors)
    outputs = Conv2D(1, (1, 1), activation='relu', name='model_output') (c7)

    # compile model here
    model = Model(inputs=[inputs], outputs=[outputs])
    # optimizer is Adam, lr is learning rate
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss=mymse, metrics=[r2_keras, mymse])
    model.summary()

    # output logs
    csv_logger = CSVLogger(path + 'DLO3_log_'+str(seed_now)+'.csv', append=True, separator=';')
    earlystopper = EarlyStopping(patience=20, verbose=1)
    checkpointer = ModelCheckpoint(path+'DLO3_checkpt_'+str(seed_now)+'.h5', verbose=1, save_best_only=True)


    results = model.fit(X_train, Y_train, validation_data=(X_test[:92*10], Y_test[:92*10]), batch_size=120, epochs=250,
                        callbacks=[earlystopper, checkpointer, csv_logger])
    model.save(path+'DLO3_VOC_'+str(seed_now)+'.h5')

    datanow = model.predict(xnow)
    np.save(path+'train_pred_'+str(seed_now)+'.npy', datanow)
    datanow = model.predict(X_test)
    np.save(path+'test_pred_'+str(seed_now)+'.npy', datanow)


if __name__ == '__main__':

    for ss in range(8):
        # choose a random seed for optimizer
        seed_now = random.randint(300,30000)
        newpath = "/some/path/to/directory/results/" + str(seed_now) + "/"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            new_model(seed_now, newpath)
        else:
            pass

