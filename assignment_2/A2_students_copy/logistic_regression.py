from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import Type

from knn import preprocess_data
from utils import x_train, y_train, x_test, y_test

np.random.seed(42)

train_images, max_val = preprocess_data(x_train)
test_images, _ = preprocess_data(x_test, max_val)

del x_train, x_test

print(f"Training inputs' shape after vectorization: {train_images.shape}")
print(f"Testing inputs' shape after vectorization: {test_images.shape}")

n_train_samples = 60000
n_val_samples = 10000
n_test_samples = 10000

# define the training set and labels
train_set = train_images[:n_train_samples]
train_labels = y_train[:n_train_samples]
print(f"Training set shape: {train_set.shape}")

# define the validation set and labels
val_set = train_images[-n_val_samples:]
val_labels = y_train[-n_val_samples:]
print(f"Validaton set shape: {val_set.shape}")

# define the test set and labels
test_set = test_images[:n_test_samples]
test_labels = y_test[:n_test_samples]
print(f"Test set shape: {test_set.shape}")

def iterate_samples(batch_size, sample_set, label_set, shuffle=True):
    # set random seed reproducibility
    np.random.seed(42)

    order = np.arange(sample_set.shape[0])
    if shuffle:
        np.random.shuffle(order)

    for i in range(0, sample_set.shape[0], batch_size):
        batch_samples = sample_set[order[i : i + batch_size]]
        batch_labels = label_set[order[i : i + batch_size]]
        yield batch_samples, batch_labels

class LogisticRegressionModel:
    def __init__(self, init_weights: np.ndarray) -> None:
        num_classes = init_weights.shape[1]
        # the weight matrix. Shape = [num_features, num_classes]
        self.W = np.copy(init_weights)
        # the bias vector. Shape = [num_classes]
        self.b = np.zeros((num_classes))

    def __call__(self, x: np.ndarray) -> np.ndarray :
        """
        Computes the hypothesis function, i.e. the prediction (y_hat) of the logistic regression model

        Args:
            x: Numpy array of shape [batch_size x num_features] containing input mini-batch of samples

        Returns:
            x: Numpy array of shape (shape: [batch_size x num_classes]) containing the output class probabilities
                after applying the softmax function
        """
        # HINT: Look into the documentation of np.matmul

        ### Implement here
        y = np.matmul(x, self.W) + self.b
        x = softmax(y)

        return x

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Takes the input and applies the softmax function to it

    Args:
        x: Numpy array

    Returns:
        np.ndarray: the softmax-ed input
    """

    ### Implement here
    x = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    return x

def nll_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Computes the negative log likelihood loss between the prediction and the target

    Args:
        prediction: Numpy array of shape [batch size x num_classes]
        target: Numpy array of shape  [batch size, ]

    Returns:
        (float): the negative log likelihood loss
    """

    batch_size = prediction.shape[0]
    loss = 0

    ### Implement here
    for i in range(batch_size):
        loss += -np.log(prediction[i, target[i]])

    loss /= batch_size

    return loss

def compute_gradients(x: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the loss function w.r.t the parameters

    Args:
        x (np.ndarray): Numpy array of shape [batch size x num_features]
        prediction (np.ndarray): Numpy array of shape [batch size x num_classes]
        target (np.ndarray): Numpy array of shape  [batch size, ]

    Returns:
        grad_W (np.ndarray): Numpy array of shape [num_features x num_classes]
                             i.e. same as the weights matrix
        grad_b (np.ndarray): Numpy array of shape [num_classes, ]
    """

    batch_size = x.shape[0]
    grad_W = np.zeros((x.shape[1], prediction.shape[1]))
    grad_b = np.zeros(prediction.shape[1])

    ### Implement here

    return grad_W, grad_b

def validation(model: Type[LogisticRegressionModel], val_set: np.ndarray, val_labels: np.ndarray,
                batch_size: int) -> float:
    """
    Performs validation of the given input model

    Args:
        model (type: class): the model to be validated
        val_set (np.ndarray): Numpy array of shape [val_size x num_features]
        val_labels (np.ndarray): Numpy array of shape [val_size]
        batch_size (int): Int defining the batch_size

    Returns:
        val_loss (float): the validation loss for the entire validation set
        val_acc (float): the validation accuracy for the entire validation set
    """

    total_loss = 0.0
    correct_preds = 0
    sample_count = 0
    batch_count = 0

    for batch, labels in iterate_samples(batch_size, val_set, val_labels, False):
        ### Implement here
        continue

    validation_loss = total_loss / batch_count
    validation_acc = correct_preds / sample_count

    return validation_loss, validation_acc

def train_one_epoch(model: Type[LogisticRegressionModel],
                    train_set: np.ndarray, train_labels: np.ndarray,
                    val_set: np.ndarray, val_labels:np.ndarray,
                    batch_size: int, learning_rate: float,
                    validation_every_x_step: int) -> float:
    """
    Trains the model for one epoch on the entire dataset with the given learning rate and batch size

    Args:
        model (class): the model used to train
        train_set (np.ndarray): Numpy array of shape [val_size x num_features]
        train_laels (np.ndarray): Numpy array of shape [val_size]
        val_set (np.ndarray): Numpy array of shape [val_size x num_features]
        val_laels (np.ndarray): Numpy array of shape [val_size]
        batch_size (int): the batch size to be used to iterate through the dataset
        learning_rate (float): the learning rate to be used for mini-batch gradient descent optimization
        validation_every_x_step (int): the number of steps to wait before performing validation

    Returns:
        train_losses (list): a list of training losses
        train_accuracies (list): a list of training accuracies
        # train_steps (list): a list of the training batch ids, i.e. each element is the n-th batch of the training set
        train_steps (list): a list of the number of training steps. One training step is defined as one forward pass
                            (i.e. calculating the loss) AND one backward pass (i.e. calculating the gradients and updating the parameters)
                            of a mini-batch of samples through the model
        val_losses (list): a list of validation losses
        val_accuracies (list): a list of validation accuracies
        val_steps (list): a list of the validation steps. One validation step is defined one forward pass of the validaton mini-batch
                            samples through the model
    """
    train_losses = []
    train_accuracies = []
    train_steps = []
    val_losses = []
    val_accuracies = []
    val_steps = []
    step_count = 0

    # Iterate through the training set and append the corresponding metrics to the list
    for x_batch, targets in iterate_samples(batch_size, train_set, train_labels, True):
        step_count += 1
        ### Implement here

        # perform validation depending on the value of `validation_every_x_step`
        if (step_count % validation_every_x_step) == 0 or step_count == 1:
            ### Implement here

            val_steps.append(step_count)

    return train_losses, train_accuracies, train_steps, val_losses, val_accuracies, val_steps

def logistic_fit_classifier(num_epochs: int, batch_size: int, learning_rate: float, validation_every_x_step: int, W_initial_weights: float) -> float:
    """
    Trains the logistic regression model

    Args:
        num_epochs (int): Number of epochs to train the model for
        batch_size (int): Size of the mini-batch
        learning_rate (float): Step size for mini-batch gradient descent optimization
        validation_every_x_step (int): Perform validation at every x-th step
        W_initial_weights (float): Randomly initialized weight matrix

    Returns:
        train_loss: list containing training losses at each epoch
        train_accuracy: list containing training accuracies at each epoch
        train_step:
        val_loss: list containing validation losses at each epoch
        val_accuracy: list containing validation accuracies at each epoch
        val_step:
    """

    train_loss = []
    train_accuracy = []
    train_step = []
    val_loss = []
    val_accuracy = []
    val_step = []
    epoch_last_step = 0

    model = LogisticRegressionModel(W_initial_weights)

    for i in tqdm(range(num_epochs)):
        epoch_train_loss, epoch_train_accuracy, epoch_train_step, \
             epoch_val_loss, epoch_val_accuracy, epoch_val_step = \
                train_one_epoch(model, train_set, train_labels, val_set,
                    val_labels, batch_size, learning_rate,
                    validation_every_x_step)
        train_loss += epoch_train_loss
        train_accuracy += epoch_train_accuracy
        train_step += [step + epoch_last_step for step in epoch_train_step]

        val_loss += epoch_val_loss
        val_accuracy += epoch_val_accuracy
        val_step += [step + epoch_last_step for step in epoch_val_step]

        epoch_last_step = train_step[-1]

    return train_loss, train_accuracy, train_step, val_loss, val_accuracy, val_step

def test_model(num_epochs: int, batch_size: int, learning_rate: float, validation_every_x_step: int, W_initial_weights: float) -> float:
    """
    Trains the logistic regression model

    Args:
        num_epochs (int): Number of epochs to train the model for
        batch_size (int): Size of the mini-batch
        learning_rate (float): Step size for mini-batch gradient descent optimization
        validation_every_x_step (int): Perform validation at every x-th step
        W_initial_weights (float): Randomly initialized weight matrix

    Returns:
        train_loss: list containing training losses at each epoch
        train_accuracy: list containing training accuracies at each epoch
        train_step:
        test_loss: list containing test losses at each epoch
        test_accuracy: list containing test accuracies at each epoch
        test_step:
    """

    train_loss = []
    train_accuracy = []
    train_step = []
    test_loss = []
    test_accuracy = []
    test_step = []
    epoch_last_step = 0

    model = LogisticRegressionModel(W_initial_weights)

    # note here that we have just replaced the validation set with the test set,
    # the rest of the procedure remains the same
    for i in tqdm(range(num_epochs)):
        epoch_train_loss, epoch_train_accuracy, epoch_train_step, \
             epoch_test_loss, epoch_test_accuracy, epoch_test_step = \
                train_one_epoch(model, train_set, train_labels, test_set,
                    test_labels, batch_size, learning_rate,
                    validation_every_x_step)

        train_loss += epoch_train_loss
        train_accuracy += epoch_train_accuracy
        train_step += [step + epoch_last_step for step in epoch_train_step]

        test_loss += epoch_test_loss
        test_accuracy += epoch_test_accuracy
        test_step += [step + epoch_last_step for step in epoch_test_step]

        epoch_last_step = train_step[-1]

    return train_loss, train_accuracy, train_step, test_loss, test_accuracy, test_step

if __name__ == "__main__":
    # initiliaze weights from a normal distribution
    W_initial_weights = np.random.normal(0.5, 0.1, (784, 15))

    # train a logistic regression model
    print("Training the model...")
    train_loss_bs100, train_accuracy_bs100, train_step_bs100, val_loss_bs100, \
        val_accuracy_bs100, val_step_bs100 = \
            logistic_fit_classifier(num_epochs=4, batch_size=100, learning_rate=0.05,
            validation_every_x_step=10, W_initial_weights=W_initial_weights)

    print(f"Train accuracy final: {train_accuracy_bs100[-1] * 100.0}, Validation accuracy: {val_accuracy_bs100[-1] * 100.0}")

    fig = plt.figure()
    plt.plot(train_step_bs100, train_accuracy_bs100, label='train_accuracy')
    plt.plot(val_step_bs100, val_accuracy_bs100, label='val_accuracy')
    plt.xlabel("Num Steps", fontsize=14)
    plt.ylabel("Val Acc.", fontsize=14)
    plt.legend()
    plt.show()


    # Uncomment the following code and add your best learning rate
    # best_batch_size = 100
    # best_lr = <BEST_LR>

    # # Test the model!
    # print("Testing the model...")
    # _ , _ , _ , \
    #     test_loss, test_accuracy, test_step = test_model(num_epochs=100, batch_size=best_batch_size,
    #                                                         learning_rate=best_lr, validation_every_x_step=100, W_initial_weights=W_initial_weights)

    # fig = plt.figure()
    # plt.plot(test_step, test_accuracy, label=f"bs={best_batch_size}, lr={best_lr}")
    # plt.title('Test Accuracy')
    # plt.ylim(0.6, 0.95)
    # plt.legend()
    # plt.show()

    # print(f"Best test accuracy: {test_accuracy[-1] * 100.0} %")