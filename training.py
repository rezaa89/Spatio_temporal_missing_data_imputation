from preprocessing import preprocessing
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

class Missing_imputation_ED():
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, labels_test_m = None,  num_features = 1, num_sensors = 1, look_back = 12, optimizer = 'adam', stats = False, loss = None):
        
        if optimizer == 'adam':
            self.optimizer = Adam(lr=0.001)#, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
        self.loss = loss
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.labels_train_m = x_train[:,:,:,1]
        self.labels_test_m = x_test[:,:,:,1]
        self.num_features = num_features
        self.num_sensors = num_sensors
        self.look_back = look_back

        if stats:
            print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    def build_fc_nn(self, img_shape):
        '''
        input: image_shape
        output: Encoder_Decoder_model , encoder_model
        
        The function builds a fully-connected neural network. All changes in the architecture of the model should be done inside this function.
        '''
        dims = [256, 512, 1024, 2048, 1024, 512, 256]; # number of hidden units.
        inputs = Input(shape=img_shape)
        inputs_ = Flatten()(inputs)
        lay_1  = Dense(units=dims[0], kernel_regularizer=regularizers.l2(0.0),kernel_initializer=he_normal(seed=0))(inputs_)
        lay_11  = LeakyReLU(alpha=0.3)(lay_1)
        lay_11 = Dropout(0.1, noise_shape=None, seed=None)(lay_11);
        lay_11 = BatchNormalization(momentum = 0.8)(lay_11)
        lay_2  = Dense(units=dims[1], kernel_regularizer=regularizers.l2(0.0),kernel_initializer=he_normal(seed=0))(lay_11)
        lay_22  = LeakyReLU(alpha=0.3)(lay_2)
        lay_22 = Dropout(0.1, noise_shape=None, seed=None)(lay_22);
        lay_22 = BatchNormalization(momentum = 0.8)(lay_22)
        lay_3  = Dense(units=dims[2], kernel_regularizer=regularizers.l2(0.0),kernel_initializer=he_normal(seed=0))(lay_22)
        lay_33  = LeakyReLU(alpha=0.3)(lay_3)
        lay_33 = Dropout(0.1, noise_shape=None, seed=None)(lay_33);
        lay_33 = BatchNormalization(momentum = 0.8)(lay_33)
        lay_4  = Dense(units=dims[3], kernel_initializer=he_normal(seed=0), 
                       kernel_regularizer=regularizers.l2(0.0), name="encoder_0")(lay_33)#activation = 'relu',
        lay_44  = LeakyReLU(alpha=0.3)(lay_4)
        lay_4d = Dropout(0.1, noise_shape=None, seed=None)(lay_44);
        lay_4d = BatchNormalization(momentum = 0.8)(lay_4d)
        lay_5  = Dense(units=dims[4], kernel_initializer=he_normal(seed=0),kernel_regularizer=regularizers.l2(0.1))(lay_4d)
        lay_55  = LeakyReLU(alpha=0.3)(lay_5)
        lay_55 = Dropout(0.1, noise_shape=None, seed=None)(lay_55);
        lay_55 = BatchNormalization(momentum = 0.8)(lay_55)
        lay_6  = Dense(units=dims[5], kernel_regularizer=regularizers.l2(0.0),kernel_initializer=he_normal(seed=0))(lay_4d)
        lay_66  = LeakyReLU(alpha=0.3)(lay_6)
        lay_66 = Dropout(0.1, noise_shape=None, seed=None)(lay_66);
        lay_66 = BatchNormalization(momentum = 0.8)(lay_66)
        lay_7  = Dense(units=dims[6], kernel_regularizer=regularizers.l2(0.0),kernel_initializer=he_normal(seed=0))(lay_4d)
        lay_77  = LeakyReLU(alpha=0.3)(lay_7)
        lay_77 = Dropout(0.1, noise_shape=None, seed=None)(lay_77);
        lay_77 = BatchNormalization(momentum = 0.8)(lay_77)
        output = Dense(units=img_shape[0]*img_shape[1]*(self.num_features+1),kernel_initializer=he_normal(seed=0), activation = 'linear',
                       name="decoder_0")(lay_77)
        output = Reshape((self.look_back, self.num_sensors, self.num_features+1))(output)
        model = Model(inputs, output)
        
        return model

    def build_cnn(self, img_shape):
        '''
        input: image_shape
        output: Encoder_Decoder_model , encoder_model
        
        The function builds a convolutional neural network. All changes in the architecture of the model should be done inside this function.
        '''
        input_img = Input(shape=(img_shape[0], img_shape[1], img_shape[2])) 
        lay_1 = Conv2D(16, activation = 'relu', strides= (1,1), kernel_size=(3, 9), padding='valid')(input_img)
        lay_1 = Dropout(0.01, noise_shape=None, seed=None)(lay_1)
        lay_1 = BatchNormalization(momentum=0.8)(lay_1)
        lay_1 = LocallyConnected2D(32, activation = 'relu', strides= (1,1), kernel_size=(3, 1), padding='valid')(lay_1)
        lay_1 = Dropout(0.0, noise_shape=None, seed=None)(lay_1)
        lay_1 = BatchNormalization(momentum=0.8)(lay_1)
        lay_1 = LocallyConnected2D(64, activation = 'relu', strides= (1,1), kernel_size=(3, 1), padding='valid')(lay_1)
        lay_1 = Dropout(0.0, noise_shape=None, seed=None)(lay_1)
        lay_1 = BatchNormalization(momentum=0.8)(lay_1)
        lay_1 = LocallyConnected2D(128, activation = 'relu', strides= (1,1), kernel_size=(3, 1), padding='valid')(lay_1)
        lay_1 = Dropout(0.0, noise_shape=None, seed=None)(lay_1)
        enc = BatchNormalization(momentum=0.8)(lay_1)
        lay_1 = Conv2DTranspose(128, strides= (1,1), kernel_size=(3, 1), padding='valid')(enc)
        lay_1 = Dropout(0.0, noise_shape=None, seed=None)(lay_1)
        lay_1 = BatchNormalization(momentum=0.8)(lay_1)
        lay_1 = Conv2DTranspose(64, strides= (1,1), kernel_size=(3, 1), padding='valid')(lay_1)
        lay_1 = Dropout(0.0, noise_shape=None, seed=None)(lay_1)
        lay_1 = BatchNormalization(momentum=0.8)(lay_1)
        lay_1 = Conv2DTranspose(32, strides= (1,1), kernel_size=(3, 1), padding='valid')(lay_1)
        lay_1 = Dropout(0.0, noise_shape=None, seed=None)(lay_1)
        lay_1 = BatchNormalization(momentum=0.8)(lay_1)
        lay_1 = Conv2DTranspose(16, strides= (1,1), kernel_size=(3, 9), padding='valid')(lay_1)
        lay_1 = Dropout(0.0, noise_shape=None, seed=None)(lay_1)
        lay_1 = BatchNormalization(momentum=0.8)(lay_1)
        lay_1 = Flatten()(lay_1)
        
        
        output = Dense(units=img_shape[0]*img_shape[1]*(self.num_features+1),kernel_initializer=he_normal(seed=0), activation = 'linear',name="decoder_0")(lay_1)
        output = Reshape((self.look_back, self.num_sensors, self.num_features+1))(output)
          
        autoencoder = Model(input_img, output)
        return autoencoder

    def build_lstm(self, img_shape):
        '''
        input: image_shape
        output: Encoder_Decoder_model , encoder_model
        
        The function builds a lstm neural network. All changes in the architecture of the model should be done inside this function.
        '''
        input_img = Input(shape=(img_shape[0], img_shape[1], img_shape[2])) 
        input_img_ = Reshape((img_shape[0], img_shape[1]*img_shape[2]))(input_img)
        lay_1, forward_h, forward_c = LSTM(256, activation = 'relu', kernel_initializer=he_normal(seed=0),
                                                                          return_sequences= True, return_state=True,)(input_img_)
        lay_1 = Dropout(0.1)(lay_1);
        lay_1 = BatchNormalization(momentum=0.8)(lay_1)
        #lay_1 = RepeatVector(img_shape[0])(lay_1)
        #encoder_output = Concatenate(axis = 1)([forward_h, forward_c, back_h, back_c])
        lay_1 = LSTM(256, activation = 'relu', kernel_initializer=he_normal(seed=0), return_sequences= False)(lay_1, [forward_h, forward_c])
        lay_1 = Dropout(0.1)(lay_1);
        lay_1 = BatchNormalization(momentum=0.8)(lay_1)
        #lay_1 = Bidirectional(LSTM(32, activation = 'relu', kernel_initializer=initializers.he_normal(seed=0), return_sequences= True))(lay_1)
        #lay_1 = Dropout(0.1, noise_shape=None, seed=None)(lay_1);
        #output = Dense(units=look_back*num_sensors*(num_features+1),kernel_initializer=initializers.he_normal(seed=0), activation = 'relu')(lay_1)
        #output = Dropout(0.01)(output)
        #output = BatchNormalization(momentum=0.8)(output)
        output = Dense(units=look_back*num_sensors*(num_features+1),kernel_initializer=he_normal(seed=0), activation = 'linear',name="decoder_0")(lay_1)
        output = Reshape((self.look_back, self.num_sensors, self.num_features+1))(output)
        
        autoencoder = Model(input_img, output)
        return autoencoder
        
    def build_bilstm(self, img_shape):
        '''
        input: image_shape
        output: Encoder_Decoder_model , encoder_model
        
        The function builds a bi-directional lstm neural network. All changes in the architecture of the model should be done inside this function.
        '''
        input_img = Input(shape=(img_shape[0], img_shape[1], img_shape[2])) 
        input_img_ = Reshape((img_shape[0], img_shape[1]*img_shape[2]))(input_img)
        lay_1, forward_h, forward_c, back_h , back_c = Bidirectional(LSTM(64, activation = 'relu', kernel_initializer=he_normal(seed=0),
                                                                          return_sequences= True, return_state=True,))(input_img_)
        lay_1 = Dropout(0.1)(lay_1);
        #lay_1 = RepeatVector(img_shape[0])(lay_1)
        #encoder_output = Concatenate(axis = 1)([forward_h, forward_c, back_h, back_c])
        lay_1 = Bidirectional(LSTM(64, activation = 'relu', kernel_initializer=he_normal(seed=0), return_sequences= True))(lay_1)
        lay_1 = Dropout(0.1)(lay_1);
        #lay_1 = Bidirectional(LSTM(32, activation = 'relu', kernel_initializer=initializers.he_normal(seed=0), return_sequences= True))(lay_1)
        #lay_1 = Dropout(0.1, noise_shape=None, seed=None)(lay_1);
        #output = Dense(units=2*img_shape[1]*num_features,kernel_initializer=initializers.he_normal(seed=0), activation = 'relu')(lay_1)
        output = Dense(units=img_shape[1]*(self.num_features+1),kernel_initializer=he_normal(seed=0), activation = 'linear',name="decoder_0")(lay_1)
        output = Reshape((self.look_back, self.num_sensors, self.num_features+1))(output)
        
        autoencoder = Model(input_img, output)
        return autoencoder
    

    def training(self, train = True, model_type = 'fcnn', img_shape = None,batch_size = 16, epochs = 10, verbose = False):
        '''
        This function builds , compile, read, save and train a neural network model.
        Arguments:
        train: if it is true, it trains the model, if it is False, it read the stored model.
        model_type: it can be fcnn, cnn, lstm, and bilstm.
        img_shape: the input size. The image-like representation of time series have a shape of n , look_back , num_sensors, num_features.
        batch_size: size of batch in traning.
        epochs: number of epochs in training.
        verbose: print out the training process with keras model.fit(.)
        Outputs:
        the function does not return any output. The function stores the neural network models.
        '''
        if model_type == 'fcnn':
            if train:
                self.model_fc_NN = self.build_fc_nn(img_shape = img_shape)
                self.model_fc_NN.compile(optimizer = MI_ED.optimizer , loss = self.loss, metrics=["mse"])
                self.model_fc_NN.summary()
                
                self.history_fc_NN = self.model_fc_NN.fit(x = self.x_train, y = self.y_train,
                                                                batch_size=batch_size, epochs = epochs, verbose = verbose)
                    
                model_json = self.model_fc_NN.to_json()
                with open("model_fcnn.json", "w") as json_file:
                    json_file.write(model_json)
                self.model_fc_NN.save_weights("model_fcnn.h5")
            else:
                json_file = open('model_fcnn.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                self.model_fc_NN = model_from_json(loaded_model_json)
                self.model_fc_NN.load_weights("model_fcnn.h5")
            return self.model_fc_NN

        if model_type == 'cnn':
            if train:
                self.model_cnn_NN = self.build_cnn(img_shape = img_shape)
                self.model_cnn_NN.compile(optimizer = MI_ED.optimizer , loss = self.loss, metrics=["mse"])
                self.model_cnn_NN.summary()
                
                self.history_cnn = self.model_cnn_NN.fit(x = self.x_train, y = self.y_train,
                                                                batch_size=batch_size, epochs = epochs, verbose = verbose)
                    
                model_json = self.model_cnn_NN.to_json()
                with open("model_cnn.json", "w") as json_file:
                    json_file.write(model_json)
                self.model_cnn_NN.save_weights("model_cnn.h5")
            else:
                json_file = open('model_cnn.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                self.model_cnn_NN = model_from_json(loaded_model_json)
                self.model_cnn_NN.load_weights("model_cnn.h5")
            return self.model_cnn_NN
                
        if model_type == 'bilstm':
            if train:
                self.model_bilstm_NN = self.build_bilstm(img_shape = img_shape)
                self.model_bilstm_NN.compile(optimizer = MI_ED.optimizer , loss = self.loss, metrics=["mse"])
                self.model_bilstm_NN.summary()
                
                self.history_bilstm = self.model_bilstm_NN.fit(x = self.x_train, y = self.y_train,
                                                                batch_size=batch_size, epochs = epochs, verbose = verbose)
                    
                model_json = self.model_bilstm_NN.to_json()
                with open("model_bilstm.json", "w") as json_file:
                    json_file.write(model_json)
                self.model_bilstm_NN.save_weights("model_bilstm.h5")
            else:
                json_file = open('model_bilstm.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                self.model_bilstm_NN = model_from_json(loaded_model_json)
                self.model_bilstm_NN.load_weights("model_bilstm.h5")
            return self.model_bilstm_NN
        

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
      
