import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from q1_1 import rmse
from q3_1 import compute_gradient_ridge, compute_gradient_simple
from q3_2 import gradient_descent_regression

# Load the dataset
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values


# Hyperparameters
num_epochs = 1000
ridge_hyperparameter = 0.1
learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # Different learning rates to try

# Provide your code here ...