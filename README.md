# Clustering of Traffic Flow Data with SpatialDeep Embedded Clustering (Keras implementation)
The project includes various neural networks for clustering of time series data.

If you found these implementatoins useful and publish a paper, please cite the following papers:

1- Asadi, Reza, and Amelia Regan. "Spatio-temporal clustering of traffic data with deep embedded clustering"; Proceedings of the 3rd ACM SIGSPATIAL International Workshop on Prediction of Human Mobility. 2019.

2- Reza Asadi; Amelia Regan. "Clustering of Traffic Flow Data with SpatialDeep Embedded Clustering"; Arxiv.

3- Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.

4- Guo, X., Zhu, E., Liu, X., & Yin, J. (2018, November). Deep embedded clustering with data augmentation. In Asian conference on machine learning (pp. 550-565).

## Data
The data is obtained from PeMS [2]. Here, we present a sample data, which only includes the traffic data for a short period of time, and few sensors in Bay Area, California. To obtain the rest of the data, you can use PeMS website.

The dataset includes traffic flow data of 6 sensors on a highway in BayArea, California. The training and testing data are each only for 10 days.

## Codes
The implementations have several steps, and each of them are described in details here. We use part of the implementation by https://github.com/XifengGuo/DEC-keras .

### Libraries
Throughout the implementation, various libraries have been used. The libraries includes numpy, pandas, matplotlib, sklearn and keras.

## Preprocessing
Preprocessing steps includes:

1- Select mainline sensors: this function gives you a list of potential sensors from PeMS. This function should be used along with PeMS website to select a subset of sensors on a given highway for a time period.

2- Read data: this function recieves the list of sensors, starting and ending date of training and testing, and it returns the dataset.

3- Sliding window method: this function receives the data and look_back variable, and it returns time series segments as data points.

4- There is a main in this file, where you can see an example of using these functions and their outputs. The functions generates data points, applies a kmeans clustering, and temporal and spatial clusters are shown in the main file.

Temporal clusters:
<p align="center">
  <img width="460" height="300" src="https://github.com/rezaa89/Spatio_temporal_missing_data_imputation/blob/master/plot_Error_missingdataratios.png">
</p>

Spatial clusters:
<p align="center">
  <img width="460" height="300" src="https://github.com/rezaa89/Spatio_temporal_missing_data_imputation/blob/master/plot_Error_missingdataratios.png">
</p>


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

## How to Run
1- Clone the project.

2- Run "python preprocessing.py" or "python training.py" or "python evaluation.py". In each of the .py files, there is a complete code of running up to the section. For example, evaluation.py includes the class and functions of Evaluation, but also call the functions of preprocessing, training and evluations in order.

## To be completed
The project consists of basic implementations of the models for spatio-temporal missing data imputation. Following parts will be added to the project:

1- Changing hyper paramters of authoencoders based on the original paper. Current models have weak performance.

2- Changing the input arguments for training.

[1] Reza Asadi, Amelia Regan, "A Convolutional Recurrent Autoencoder for Spatio-temporal Missing Data Imputation", Int'l Conf. Artificial Intelligence | ICAI 2019

[2] “California. pems, http://pems.dot.ca.gov/, 2017,”
