# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 23:40:19 2020

@author: Reza
"""

from preprocessing import preprocessing
from evaluation import Evaluation
from evaluation import Missing_imputation_ED

from scipy.ndimage.interpolation import shift
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Concatenate,BatchNormalization,Conv2DTranspose, LeakyReLU, ConvLSTM2D, Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import Conv2D, Conv2DTranspose, LocallyConnected2D, LSTM, Bidirectional, TimeDistributed , RepeatVector, Reshape, Add, Concatenate, BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.models import model_from_json, load_model, Model
from keras.losses import mse

from keras import backend as K
from keras.optimizers import Adam
from keras.backend import int_shape

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.advanced_activations import LeakyReLU
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from random import sample, choices

from keras import regularizers
from keras.initializers import he_normal
import math 
import matplotlib.pyplot as plt


    

if __name__ == '__main__':
      p_obj = preprocessing()
      mainline_list = p_obj.select_mainline_sensors()
      dataset_train, dataset_test, scaler_alldata = p_obj.read_data()
      X_train_nrescaled , X_test_nrescaled, X_train_ , X_test_ = p_obj.sliding_window(dataset_train.values, dataset_test.values)
      
      
      num_augmentations = 15

      x_train_rep = np.tile(X_train_, (num_augmentations,1,1,1))

      x_train_input, labels_train_aug = p_obj.augment_rand(x_train_rep, l_percent = 0.1, u_percent = 0.2, weighted = False, noise = False)

      alpha = 0

      mask = (1-labels_train_aug)+alpha
      #print(x_train_augmentation.shape)
      
      X_train_target = np.concatenate((x_train_rep,mask), axis = 3)
      
      #del x_train_augmentation, labels_train_aug

      x_test_m, labels_test_m = p_obj.augment_rand(X_test_, l_percent = 0.1, u_percent = 0.2, weighted = False, noise = False)

      
      json_file = open('model_cnn_bilstm.json', 'r')
      loaded_model_json = json_file.read()
      json_file.close()
      model = model_from_json(loaded_model_json)
      model.load_weights("model_cnn_bilstm.h5")
      
      eval_obj = Evaluation()
      rmse_list = []
      mae_list = []
      for i in np.arange(0.0, 1.0,0.1):
        x_test_input, labels_test_ = p_obj.augment_rand(X_test_, l_percent = i, u_percent = i+0.02)
        #print(i, np.sum(1-labels_test_aug)/(labels_test_aug.shape[0]*labels_test_aug.shape[1]*labels_test_aug.shape[2]))
        p_train_ave, p_test_ave = eval_obj.generates_predictions(x_train = X_train_, x_test = x_test_input, scaler_alldata = scaler_alldata, model = model, train = False, test = True)
        rmse_ , mae_ = eval_obj.evaluations(X_train_nrescaled, X_test_nrescaled, p_train_ave, p_test_ave, labels_train_aug, labels_test_, p_obj.num_sensors, train = False, test = True)
        rmse_list += [rmse_]
        mae_list += [mae_]
      eval_obj.plot_(rmse_list, mae_list)
