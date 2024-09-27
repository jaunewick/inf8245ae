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


np.random.seed(42)  # For reproducibility
n_features = X_train.shape[1]
initial_w = np.random.normal(0, 1, size=n_features)
initial_b = 0.0

learning_rate = 0.00005  # You can change this value to get better results
num_epochs = 1000
ridge_hyperparameter = 20 # You can change this value to get better results

# Provide your code here ...
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)
y_train = y_train.flatten()

training_losses_simple = []
training_losses_ridge = []

for epoch in range(num_epochs):
    w_simple, b_simple = gradient_descent_regression(
        X_train, y_train,
        reg_type='simple',
        learning_rate=learning_rate,
        num_epochs=epoch
    )
    y_hat_simple = np.dot(X_train, w_simple) + b_simple
    training_loss = rmse(y_train, y_hat_simple)
    training_losses_simple.append(training_loss)

    w_ridge, b_ridge = gradient_descent_regression(
        X_train, y_train,
        reg_type='ridge',
        hyperparameter=ridge_hyperparameter,
        learning_rate=learning_rate,
        num_epochs=epoch
    )
    y_hat_ridge = np.dot(X_train, w_ridge) + b_ridge
    training_loss = rmse(y_train, y_hat_ridge)
    training_losses_ridge.append(training_loss)

plt.plot(training_losses_simple, color='blue', alpha=0.5, label='Simple Linear Regression')
plt.plot(training_losses_ridge, color='red', alpha=0.5, label=f'Ridge Regression (λ = {ridge_hyperparameter})')
plt.title('Training Loss vs. Epoch')
plt.ylabel('Training Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

