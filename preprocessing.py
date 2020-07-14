# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 00:50:39 2020

@author: Reza
"""

# preprocessing file.

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from random import sample
from random import choices
import random
from datetime import datetime

class preprocessing():
  def __init__(self, stations = [403401 ,403409 ,403412 ,402067 ,403414 ,403419 ,401163 ,400823 ,401846], look_back= 12):
    '''
    This class read and preprocess traffic flow data.
    Arguments:
    stations = based on the sensor_ids from PeMS website you should obtain this list.
    look_back = the size of look_back variable in sliding window method.
    '''
    self.stations = stations
    self.look_back = look_back
    self.num_sensors = len(stations)

  def select_mainline_sensors(self, list_stations = './total_stations_BayArea.csv', fwy_name = 'I280-S', sensor_type = 'Mainline'):
    '''
    This function reads the list of stations, receives freeway and type of sensors, and return the ID of selected sensors.
    Arguments:
    list_stations = list of stations obtained from PeMS website.
    fwy_name = the freeway name. From the list of station, it should be selected.
    sensor_type = it can be selected from the list_stations and type column.
    Outputs:
    mainline_list = list of sensor ids in the dataset. The values are between 0 to the number of sensors.
    '''
    ds_s = pd.read_csv(list_stations, index_col = 0)
    ds_s = ds_s[ds_s.ID.isin(set(self.stations))]
    ds_s.index = range(ds_s.shape[0])
    self.mainline_list = ds_s[(ds_s['Fwy'] == fwy_name) & (ds_s['Type']==sensor_type)].index.tolist()
    return self.mainline_list

  def read_data(self, dataset = './dataset.csv', statistics = False, rolling_average = 0, flow = True, speed = False, occupancy = False, s1 = '02/01/2016', e1 = '03/01/2016', s2 = '03/01/2016', e2 = '04/01/2016'):
    dataset = pd.read_csv(dataset,index_col = 0)
    dataset.index = pd.to_datetime(dataset.index)
    cols_flow_m = list();
    self.num_features = sum([flow, occupancy, speed])
    for x in self.mainline_list:
        # in this project we only impute flow data. It is possible to do it for speed and occupancy.
        if flow:
          cols_flow_m.append(3*x); #flow
        if occupancy:
          cols_flow_m.append(3*x+1); #occupancy
        if speed:
          cols_flow_m.append(3*x+2); #speed
    dataset = dataset.iloc[:, cols_flow_m]
    
    if rolling_average>0:
        dataset = dataset.rolling(rolling_average)
        dataset = dataset.mean()
    
    dataset = dataset.fillna(method = 'bfill')
    dataset = dataset.fillna(method = 'ffill')

    dataset_train = dataset[(dataset.index>=s1) & (dataset.index<e1)]

    self.scaler_alldata = MinMaxScaler(feature_range=(0, 1));
    self.scaler_alldata.fit(dataset_train);

    dataset_test = dataset[(dataset.index<e2) & (dataset.index>=s2)]
    if statistics:
        print('total dataset Size : ', dataset.shape, ' min time: ', dataset.index.min(),' max time: ', dataset.index.max(), ' date range 5 min steps: ' , pd.date_range('2016-01-01 00:00:00', '2016-07-13 23:55:00', freq = '5min').shape )
        print('train dataset size: ', dataset_train.shape, ' min time: ', dataset_train.index.min(),' max time: ', dataset_train.index.max(), ' date range 5 min steps: ' , pd.date_range('2016-01-01 00:00:00', '2016-04-30 23:55:00', freq = '5min').shape)
        print('test dataset size: ', dataset_test.shape, ' min time: ', dataset_test.index.min(),' max time: ', dataset_test.index.max(), ' date range 5 min steps: ' , pd.date_range('2016-05-01 00:00:00', '2016-06-30 23:55:00', freq = '5min').shape)
    return dataset_train, dataset_test, self.scaler_alldata
  
  def sliding_window(self, dataset_train, dataset_test):
    '''
    Generates data points of a time series with sliding window method. 
    Arguments: (t*s*f)
    dataset_train: training data with size of t * s * f, t=total time stamps, s = number of sensors, f = number of features
    dataset_test: testing data
    Outputs: (look_back * s * f)
    dataset_train_nr: datapoints of training data without rescaled.
    dataset_test_nr: datapoints of testing data without rescaled.
    dataset_train: datapoints of training data
    dataset_test: datapoints of testing data
    '''
    X_train = list();
    X_test = list();
    X_train_nr = list()
    X_test_nr = list()

    for ind in range(0, dataset_train.shape[0]-self.look_back): # for each time window
        a = dataset_train[ind:ind+self.look_back,:]
        input_ = np.reshape(a, (self.look_back*self.num_sensors*self.num_features));
        X_train_nr.append(input_);
        
        a = self.scaler_alldata.transform(dataset_train[ind:ind+self.look_back,:])
        input_ = np.reshape(a, (self.look_back*self.num_sensors*self.num_features));
        X_train.append(input_);
        

    for ind in range(0, dataset_test.shape[0]-self.look_back):
        a = dataset_test[ind:ind+self.look_back,:]
        input_ = np.reshape(a, (self.look_back*self.num_sensors*self.num_features));
        X_test_nr.append(input_);
        
        a = self.scaler_alldata.transform(dataset_test[ind:ind+self.look_back,:])
        input_ = np.reshape(a, (self.look_back*self.num_sensors*self.num_features));
        X_test.append(input_);
        
    
    X_train_nr = np.asarray(X_train_nr);
    X_test_nr = np.asarray(X_test_nr);


    X_train = np.asarray(X_train);
    X_test = np.asarray(X_test);

    X_train_nr = np.reshape(X_train_nr, (X_train_nr.shape[0], self.look_back,self.num_sensors,self.num_features))
    X_test_nr = np.reshape(X_test_nr, (X_test_nr.shape[0], self.look_back,self.num_sensors,self.num_features))
    
    X_train = np.reshape(X_train, (X_train.shape[0], self.look_back,self.num_sensors,self.num_features))
    X_test = np.reshape(X_test, (X_test.shape[0], self.look_back,self.num_sensors,self.num_features))
    
    return X_train_nr , X_test_nr , X_train , X_test

  def augment_rand(self, x, l_percent = 0.1, u_percent = 0.2, weighted = False, noise = False):
    '''
    This function generates random missing values.
    Arguments:
    x: input data in the form of (n*k) * (t*s*f) , n is the total number of data points generated in sliding window method, and k is the augmentation repeatition in the data.
    l_percent: minimum percentage of missing values in each data point.
    u_percent: maximum percentage of missing values in each data point.
    weighted: it adds more weights to percentage of missing values.
    noise: it adds noise to the healthy values in the training data to better augment new data points.
    Outputs:
    x: training data with missing values.
    labels_: 0=missing values, 1=healthy values. 
    '''
    labels_ = np.ones_like(x)
    
    for ind in range(x.shape[0]):
      if weighted:
        percent = choices(np.arange(0.01,1,0.05), [5,3,3,2,2,2,2,1,1,1,1,1,1,2,2,2,3,3,3,5])[0]
      else:
        percent = random.random()*(u_percent - l_percent) + l_percent  
      percent_int = (int)(percent*x.shape[1]*x.shape[2])
      sequence = [i for i in range(x.shape[1]*x.shape[2])]
      selected_indices = sample(sequence,percent_int)    
      selected_indices_noise = sample(sequence,percent_int//10) 
      #print(len(selected_indices_noise), len(selected_indices))
      xx = np.ones((x.shape[1]*x.shape[2],))
      xx[selected_indices] = 0
      if noise:
        a = np.reshape(x[ind], (12*9,1))
        a[selected_indices_noise] =a[selected_indices_noise]*(np.random.rand(len(selected_indices_noise),1)*0.06+0.97)
        x[ind] = np.reshape(a, (12,9,1))
      xx = np.reshape(xx,(x.shape[1],x.shape[2],1))
      labels_[ind] = xx
    labels_ = np.reshape(labels_, (labels_.shape[0],labels_.shape[1],labels_.shape[2], self.num_features))
    x = np.reshape(x, (x.shape[0],x.shape[1],x.shape[2], self.num_features))
    x = np.concatenate((x*labels_,labels_), axis = 3)
    return x, labels_



if __name__ == '__main__':
  
  import argparse

  parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--list_station_filename', default='./total_stations_BayArea.csv') #, choices=['mnist', 'usps', 'reutersidf10k']
  parser.add_argument('--fwy_name', default='I280-S', type=str)
  parser.add_argument('--sensor_type', default='Mainline', choices=['Mainline', 'Off-ramp', 'on-ramp'])

  parser.add_argument('--dataset', default='./dataset.csv', type=str)
  parser.add_argument('--start_train', default='02/01/2016', type=str)
  parser.add_argument('--start_test', default='03/01/2016', type=str)
  parser.add_argument('--end_train', default='03/01/2016', type=str)
  parser.add_argument('--end_test', default='04/01/2016', type=str)

  parser.add_argument('--is_flow', default=True, type=bool)
  parser.add_argument('--is_speed', default=False, type=bool)
  parser.add_argument('--is_occupancy', default=False, type=bool)
  parser.add_argument('--show_data_stats', default=True, type=bool)

  parser.add_argument('--num_augmentations', default=15, type=int)


  parser.add_argument('--weighted_aug', default=False, type=bool)
  parser.add_argument('--noisy_args', default=False, type=bool)

  parser.add_argument('--low_percentage_missing_train', default=0.01, type=float)
  parser.add_argument('--high_percentage_missing_train', default=0.98, type=float)

  parser.add_argument('--low_percentage_missing_test', default=0.01, type=float)
  parser.add_argument('--high_percentage_missing_test', default=0.98, type=float)
  
  parser.add_argument('--weighted_loss_alpha', default=0.0, type=float)
  
  
  args = parser.parse_args()

  p_obj = preprocessing()
  mainline_list = p_obj.select_mainline_sensors(list_stations = args.list_station_filename, fwy_name = args.fwy_name, sensor_type = args.sensor_type)
  dataset_train, dataset_test, scaler_alldata = p_obj.read_data(dataset = args.dataset, statistics = args.show_data_stats, rolling_average = 0, flow = args.is_flow, 
                                                                speed = args.is_speed, occupancy = args.is_occupancy, s1 = args.start_train, e1 = args.end_train, s2 = args.start_test, e2 = args.end_test)
  X_train_nrescaled , X_test_nrescaled, X_train_ , X_test_ = p_obj.sliding_window(dataset_train.values, dataset_test.values)
  
  
  x_train_rep = np.tile(X_train_, (args.num_augmentations,1,1,1))

  x_train_input, labels_train_aug = p_obj.augment_rand(x_train_rep, l_percent = args.low_percentage_missing_train, u_percent = args.high_percentage_missing_train, weighted = args.weighted_aug, noise = args.noisy_args)

  

  mask = (1-labels_train_aug) + labels_train_aug*args.weighted_loss_alpha
  X_train_target = np.concatenate((x_train_rep,mask), axis = 3)

  
  #del x_train_augmentation, labels_train_aug
  x_test_m, labels_test_m = p_obj.augment_rand(X_test_, l_percent = args.low_percentage_missing_test, u_percent = args.high_percentage_missing_test, weighted = False, noise = False)
