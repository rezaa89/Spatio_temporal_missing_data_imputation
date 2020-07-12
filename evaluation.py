from preprocessing import preprocessing
from training import Missing_imputation_ED


import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from random import sample, choices

import math 
import matplotlib.pyplot as plt



class Evaluation():
    def generates_predictions(self, x_train , x_test = None ,scaler_alldata = None, model = None ,train = False, test = True):
        '''
        The function receives training and testing data and predicts their outputs.
        Arguments:
        x_train: training data
        x_test: testing data.
        scaler_alldata: the fitted scaller on training data to rescale the values from range of [0,1] to the original values.
        model_type: the model type to predicts the output.
        train: if the model predicts the output for training data.
        test: if the model predicts the output for testing data.
        Outputs:
        the model outputs predictions (or imputation of missing values) and returns p_train_ave or p_test_ave.
        '''
        
        if train:
          p_train = model.predict(x_train)
        if test:
          p_test = model.predict(x_test)

        p_train_ave = []
        p_test_ave = []
        
        
        if train:
            p_train = np.reshape(p_train, (p_train.shape[0], p_train.shape[1], p_train.shape[2]))
            for ind in range(x_train.shape[0]):
                p_train_ave += [scaler_alldata.inverse_transform(p_train[ind])]
            p_train_ave = np.asarray(p_train_ave)
        if test:
            p_test = np.reshape(p_test[:,:,:,0], (p_test.shape[0], p_test.shape[1], p_test.shape[2]))
            for ind in range(x_test.shape[0]):
                p_test_ave += [scaler_alldata.inverse_transform(p_test[ind])]
            p_test_ave = np.asarray(p_test_ave)
        return p_train_ave, p_test_ave
     
    def evaluations(self, dataset_train, dataset_test, p_train_ave, p_test_ave, labels_train, labels_test, num_sensors, train = False, test = True):
        '''
        This function is for evaluation of missing data imputation models.
        '''
        rmse_train = 0
        mae_train = 0 
        rmse_test = 0
        mae_test = 0 
        
        if train:
          y_train = np.reshape(dataset_train, (dataset_train.shape[0], dataset_train.shape[1]*dataset_train.shape[2]))
          p_train_ave = np.reshape(p_train_ave, (p_train_ave.shape[0], p_train_ave.shape[1]*p_train_ave.shape[2]))
          labels_train = np.reshape(labels_train, (labels_train.shape[0], labels_train.shape[1]*labels_train.shape[2]))
        
        if test:
          y_test = np.reshape(dataset_test, (dataset_test.shape[0], dataset_test.shape[1]*dataset_test.shape[2]))
          p_test_ave = np.reshape(p_test_ave, (p_test_ave.shape[0], p_test_ave.shape[1]*p_test_ave.shape[2]))
          labels_test = np.reshape(labels_test, (labels_test.shape[0], labels_test.shape[1]*labels_test.shape[2]))

        def rmse(x,y):
            return math.sqrt( np.sum((x - y)**2)*(1/x.shape[0]))

        def mae(x,y):
            return np.sum(np.abs(x - y))*(1/x.shape[0])
        
        for ind in range(y_test.shape[1]):
            if train:
              rmse_train += rmse(y_train[:,ind], p_train_ave[:,ind])
              mae_train += mae(y_train[:,ind], p_train_ave[:,ind]) 
            rmse_test += rmse(y_test[:,ind], p_test_ave[:,ind]) 
            mae_test += mae(y_test[:,ind], p_test_ave[:,ind]) 
            #print(rmse_train)
        print('Errors for all data')
        print('RMSE train: {0:.2f}, RMSE test: {1:.2f}, MAE train: {2:.2f}, MAE test:  {3:.2f}'.format( 
              rmse_train/y_test.shape[1], rmse_test/y_test.shape[1],mae_train/y_test.shape[1], mae_test/y_test.shape[1]))
        
        rmse_train = 0
        mae_train = 0 
        rmse_test = 0
        mae_test = 0 

        def rmse_m(x,y,l):
          if np.sum(l) == 0:
            return 0
          return math.sqrt( np.sum((x*l - y*l)**2)*(1/np.sum(l)))

        def mae_m(x,y,l):
          if np.sum(l) == 0:
            return 0
          return np.sum(np.abs(x*l - y*l))*(1/np.sum(l))
        
        for ind in range(y_test.shape[1]):
            if train:
              rmse_train += rmse_m(y_train[:,ind], p_train_ave[:,ind], 1-labels_train[:,ind])
              mae_train += mae_m(y_train[:,ind], p_train_ave[:,ind], 1-labels_train[:,ind]) 
            rmse_test += rmse_m(y_test[:,ind], p_test_ave[:,ind], 1-labels_test[:,ind]) 
            mae_test += mae_m(y_test[:,ind], p_test_ave[:,ind], 1-labels_test[:,ind]) 
            #print(rmse_train)
          
        print('Errors for missing data')
        print('RMSE train: {0:.2f}, RMSE test: {1:.2f}, MAE train: {2:.2f}, MAE test:  {3:.2f}'.format( 
              rmse_train/y_test.shape[1],rmse_test/y_test.shape[1],mae_train/y_test.shape[1], mae_test/y_test.shape[1]))
        return rmse_test/y_test.shape[1], mae_test/y_test.shape[1]

    def plot_(self, rmse_list, mae_list):
        '''
        The function plots the RMSE and MAE of missing data imputation models.
        '''
        N = len(rmse_list)
        ind = np.arange(N) 
        width = 0.27      

        fig = plt.figure(figsize=(16,12))
        ax = fig.add_subplot(111)

        yvals = rmse_list
        rects1 = ax.bar(ind, yvals, width, color='r', hatch='/')
        zvals = mae_list
        rects2 = ax.bar(ind+width, zvals, width, color='b', hatch='\\')

        #ax.set_title('Convergence of cyber nodes optimum solution', fontsize=20)
        ax.set_ylabel('Error', fontsize=20)
        #ax.set_xlabel('Models', fontsize=20)
        ax.set_xticks(ind+width)
        ax.set_yticks( np.arange(0,max(rmse_list)+5,5))
        ax.set_yticklabels( np.arange(0,max(rmse_list)+5,5), fontsize=12 )
        ax.set_xticklabels( range(1, 100, 5), fontsize=18 )
        ax.legend( (rects1[0], rects2[0]), ('RMSE', 'MAE'),fontsize =20 )
        plt.show()
    
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

      MI_ED = Missing_imputation_ED(x_train = x_train_input, y_train = X_train_target, x_test = x_test_m, y_test = X_test_, num_features = p_obj.num_features, num_sensors = p_obj.num_sensors, look_back = p_obj.look_back, stats = True, loss = mse)
      model = MI_ED.training(train = True,  model_type = 'bilstm', img_shape = (p_obj.look_back, p_obj.num_sensors, 2), batch_size = 512, epochs = 1, verbose = True)
      
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
