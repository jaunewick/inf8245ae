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
X_train_bias = data_matrix_bias(X_train)
w_star = linear_regression_optimize(y_train, X_train_bias)

X_test_with_bias = data_matrix_bias(X_test)
y_hat = linear_regression_predict(X_test_with_bias, w_star)

# Report the RMSE and Plot the data on the test set
rmse_value = rmse(y_test, y_hat)
print(f'RMSE value : {rmse_value}')

experience_col = X_test[:,0]
test_score_col = X_test[:,1]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(experience_col, y_test, color='blue', alpha=0.5, label='Actual Salary')
plt.scatter(experience_col, y_hat, color='red', alpha=0.5, label='Predicted Salary')
plt.title('Experience vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(test_score_col, y_test, color='blue', alpha=0.5, label='Actual Salary')
plt.scatter(test_score_col, y_hat, color='red', alpha=0.5, label='Predicted Salary')
plt.title('Test Score vs Salary')
plt.xlabel('Test Score')
plt.ylabel('Salary')
plt.legend()

plt.tight_layout(w_pad=4)
plt.show()
