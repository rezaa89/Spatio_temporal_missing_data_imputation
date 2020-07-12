# Spatio-temporal missing data imputation with Autoencoders (Keras implementation)
The project includes various autoencoder neural networks for spatio-temporal missing data imputation in Keras.

If you found these implementatoins useful, please cite the following paper:

Reza Asadi, Amelia Regan, "A Convolutional Recurrent Autoencoder for Spatio-temporal Missing Data Imputation", Int'l Conf. Artificial Intelligence | ICAI 2019.
https://csce.ucmss.com/cr/books/2019/LFS/CSREA2019/ICA2374.pdf [1]

## Data
The data is obtained from PeMS [2]. Here, we present a sample training and testing, while to use such a data, we recommend you can use it from PeMS website.

The dataset includes traffic flow data of 9 sensors on a highway in BayArea, California. We only include healthy data, and for 1 month of training and 1 month of testing data.

## Preprocessing
Preprocessing steps includes:

1- Scaling each sensor's value in range of 0-1. 
2- Applying sliding window method to obtain data points. A data point is a matrix of size (look_back*num_sensors).
3- The method generates random missing values. The input data includes missing values. The target does not have missing values. The labels are 0: missing value, 1: healthy values.

## Training
We implement following models:
1- A baseline which is hourly-weekly lookup table. 
2- Fully-connected neural network.
3- Convolutional neural network.
4- Bi-directional LSTM.

## Evaluation
The evaluation metrics are RMSE and MAE of missing data imputations. 

<p align="center">
  <img width="460" height="300" src="https://github.com/rezaa89/Spatio_temporal_missing_data_imputation/blob/master/plot_Error_missingdataratios.png">
</p>

## To be completed
The project consists of basic implementations of the models for spatio-temporal missing data imputation. Following parts will be added to the project:

1- Change of hyper paramters of authoencoders based on the original paper. Current models have weak performance.

2- Changing the input arguments for training.

[1] Reza Asadi, Amelia Regan, "A Convolutional Recurrent Autoencoder for Spatio-temporal Missing Data Imputation", Int'l Conf. Artificial Intelligence | ICAI 2019
[2] “California. pems, http://pems.dot.ca.gov/, 2017,”
