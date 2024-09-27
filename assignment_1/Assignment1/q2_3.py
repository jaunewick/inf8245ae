import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from q2_2 import cross_validation_linear_regression
from q1_1 import data_matrix_bias

# Define a range of alpha values for hyperparameter search
hyperparams = np.logspace(-4, 4, 50)
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values
kfolds = 5

# Write your code here ...
X_train_bias = data_matrix_bias(X_train)
X_test_bias = data_matrix_bias(X_test)

best_hyperparam, best_mean_squared_error, mean_squared_errors_list = cross_validation_linear_regression(
    kfolds,
    hyperparams,
    X_train_bias, y_train
)

print(f'Best hyperparameter λ value : {best_hyperparam}')
print(f'Best RMSE value : {best_mean_squared_error}')

best_lambda_index = hyperparams.tolist().index(best_hyperparam)

plt.figure(figsize=(12, 10))
plt.plot(hyperparams, mean_squared_errors_list, color='red', alpha=0.5, label='RMSE')
plt.title('RMSE vs. Ridge Regression hyperparameter λ value')
plt.ylabel('RMSE')
plt.xlabel('hyperparamter λ value')
plt.show()