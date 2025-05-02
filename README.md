# Machine Learning Models for Genomic Prediction
# Overview
This repository contains multiple machine learning (ML) and deep learning (DL) scripts, covering various algorithms and techniques. A test file is provided for users to run and validate these scripts.

# Requirements
- Python 3 or higher
- Numpy
- Pandas
- Scikit-learn
- Scipy
- TensorFlow
- Keras
- Matplotlib
- GPU (For Deep Learning programs)

NOTE:- User can run deep learning programs in CPU by adding the following line in the python scripts
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Download files
- Adaboost.py
- ANN.py
- Bagging.py
- BiLSTM.py
- cboost.py
- CNN.py
- DNN.py
- elm.py
- GRU.py
- LSTM.py
- MLP.py
- RF.py
- RNN.py
- SVR.py
- XGBoost.py
- test.csv

# File description
## Machine Learning 
- Adaboost.py = AdaBoost (Adaptive boosting) regressor
- Bagging.py = Bagging (Bootstrap aggregating) regressor
- cboost.py = CatBoost (Categorical Boosting) regressor
- elm.py = Extreme Learning Machine for regression problems
- RF.py = Random forest regressor
- SVR.py = Support Vector Regression
- XGBoost.py = Extreme Gradient Boosting regressor

## Deep Learning
- ANN.py = Artificial neural network
- BiLSTM.py = Bidirectional Long Short-Term Memory
- CNN.py = Convolutional neural network
- DNN.py = Dense neural network
- GRU.py = Gated Recurrent Unit Networks
- LSTM.py = Long Short-Term Memory
- MLP.py = Multi layer perceptron
- RNN.py = Recurrent neural network

## Input file
- Genotypic file in csv format without header with associated phenotypic values on the last column

  ### Example file
- test.csv = genotypic file with associated phenotypic values on the last column

# Usage 
- Create a working directory 
- Place the python scripts for machine learning and deep learning algorithms in the working directory along with the input file
- Running script

      python <script.py> <input_file.csv>
   
  Example:
   
      python RF.py test.csv

# Output description
When the script is executed, it performs 5-fold cross-validation (5 repetitions of 5-fold CV) on the given dataset. A result directory is created and the results are stored in csv files for further analysis. Each csv file contains four columns for each fold representing following:
- Mean Square Error (MSE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Correlation values

### User can compute the average of the 5-fold values for each evaluation metric from the csv files by running following command in the Linux shell

    awk '{a+=$1} {b+=$2} {c+=$3} {d+=$4} END {printf "%.3f\t%.3f\t%.3f\t%.3f\n", a/5, b/5, c/5, d/5}' iteration_1.csv
   
NOTE:- User can setup parameters for cross validation according to there requirement by changing the number of n_splits for number of folds and n_repeats for number of iterations in the script. By default number of n_splits and n_repeats is 5.
