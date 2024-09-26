import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from q1_1 import data_matrix_bias, linear_regression_predict, linear_regression_optimize, rmse


# Loading the dataset
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values

# Write your code here:

# Find the optimal parameters only using the training set

# Report the RMSE and Plot the data on the test set
