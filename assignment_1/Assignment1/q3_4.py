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
learning_rates = [0.00003, 0.00004, 0.000045, 0.00005, 0.000055, 0.00006, 0.00008, 0.0001]  # Different learning rates to try

# Provide your code here ...
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)
y_train = y_train.flatten()
y_test = y_test.flatten()

test_rmse_simple = []
test_rmse_ridge = []

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for i, learning_rate in enumerate(learning_rates):
    np.random.seed(42)
    n_features = X_train.shape[1]
    initial_w = np.random.normal(0, 1, size=n_features)
    initial_b = 0.0

    training_losses_simple = []
    training_losses_ridge = []

    w_simple, b_simple = initial_w, initial_b
    w_ridge, b_ridge = initial_w, initial_b

    for epoch in range(num_epochs):
        w_simple, b_simple = gradient_descent_regression(
            X_train, y_train,
            reg_type='simple',
            learning_rate=learning_rate,
            num_epochs=epoch
        )
        y_hat_train = np.dot(X_train, w_simple) + b_simple
        training_loss = rmse(y_train, y_hat_train)
        training_losses_simple.append(training_loss)

        w_ridge, b_ridge = gradient_descent_regression(
            X_train, y_train,
            reg_type='ridge',
            hyperparameter=ridge_hyperparameter,
            learning_rate=learning_rate,
            num_epochs=epoch
        )
        y_hat_train = np.dot(X_train, w_ridge) + b_ridge
        training_loss = rmse(y_train, y_hat_train)
        training_losses_ridge.append(training_loss)

    y_hat_test_simple = np.dot(X_test, w_simple) + b_simple
    test_loss_simple = rmse(y_test, y_hat_test_simple)
    test_rmse_simple.append(test_loss_simple)

    y_hat_test_ridge = np.dot(X_test, w_ridge) + b_ridge
    test_loss_ridge = rmse(y_test, y_hat_test_ridge)
    test_rmse_ridge.append(test_loss_ridge)

    ax = axes[i // 4, i % 4]
    ax.plot(training_losses_simple, color='blue', alpha=0.5, label='Simple Linear Regression')
    ax.plot(training_losses_ridge, color='red', alpha=0.5, label=f'Ridge Regression (λ = {ridge_hyperparameter})')
    ax.set_title(f'Training Loss vs. Epoch \n(With Learning Rate = {learning_rate})')
    ax.set_ylabel('Training Loss')
    ax.set_xlabel('Epoch')
    ax.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
plt.plot(learning_rates, test_rmse_simple, color='blue', alpha=0.5, label='Simple Linear Regression')
plt.plot(learning_rates, test_rmse_ridge, color='red', alpha=0.5, label=f'Ridge Regression (λ = {ridge_hyperparameter})')
plt.title('RMSE vs. Learning Rate on the test dataset')
plt.ylabel('RMSE')
plt.xlabel('Learning Rate')
plt.legend()
plt.show()