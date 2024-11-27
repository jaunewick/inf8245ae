import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from typing import List, Tuple, Union
from tqdm.auto import tqdm
from q1 import Layer, Dense, SoftmaxLayer, ReLULayer, CrossEntropyLossLayer
from q2 import MLP

## SET GLOBAL SEED
## Do not modify this for reproducibility
np.random.seed(33)

# Dataset loader
class FashionMNIST:
    def __init__(self, batch_size, val_perc: float = 0.2):
        self.data_seed = 6666
        np.random.seed(self.data_seed)
        torch.manual_seed(self.data_seed)
        self.generator = torch.Generator().manual_seed(self.data_seed)

        self.batch_size = batch_size
        self.val_perc = val_perc
        self.load_data()

    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])
        self.train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        if self.val_perc and self.val_perc > 0:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset,
                [
                    int((1 - self.val_perc) * len(self.train_dataset)),
                    int(self.val_perc * len(self.train_dataset))
                ],
                generator=self.generator
            )
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=4, pin_memory=True,
                generator=self.generator
            )
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
            generator=self.generator
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
            generator=self.generator
        )

# PyTorch model
def create_pytorch_model(hidden1, hidden2=None):
    if hidden2 is None:
        return nn.Sequential(
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, 10),
            nn.Softmax(dim=1)
        )
    else:
        return nn.Sequential(
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 10),
            nn.Softmax(dim=1)
        )

# Custom model
def create_custom_model(hidden1, hidden2=None):
    if hidden2 is None:
        return MLP([
            Dense(28 * 28, hidden1),
            ReLULayer(),
            Dense(hidden1, 10),
            SoftmaxLayer()
        ])
    else:
        return MLP([
            Dense(28 * 28, hidden1),
            ReLULayer(),
            Dense(hidden1, hidden2),
            ReLULayer(),
            Dense(hidden2, 10),
            SoftmaxLayer()
        ])

# Training loop
def run_experiment(configurations, dataset, model_type="pytorch", lr=1e-2, epochs=20):
    results = []
    for config in configurations:
        hidden1, hidden2 = config
        print(f"Running {model_type} with hidden sizes: {hidden1}, {hidden2}")

        if model_type == "pytorch":
            model = create_pytorch_model(hidden1, hidden2)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
        else:
            model = create_custom_model(hidden1, hidden2)
            criterion = CrossEntropyLossLayer()

        start_time = time.time()
        train_losses = []
        val_accuracies = []
        for _ in range(epochs):
            running_loss = 0.0
            correct, total = 0, 0

            # Training
            for images, labels in dataset.train_loader:
                images = images.view(-1, 28 * 28)
                if model_type == "pytorch":
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels.numpy())
                    model.backward(criterion.backward(loss))
                    model.update(lr)
                running_loss += loss.item()

            # Validation
            if dataset.val_perc and dataset.val_perc > 0:
                correct, total = 0, 0
                with torch.no_grad():
                    for images, labels in dataset.val_loader:
                        images = images.view(-1, 28 * 28)
                        outputs = model(images)
                        predicted = np.argmax(outputs, axis=1) if model_type == "custom" else torch.argmax(outputs, dim=1)
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
                val_accuracies.append(correct / total)

            train_losses.append(running_loss / len(dataset.train_loader))

        # Test accuracy
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in dataset.test_loader:
                images = images.view(-1, 28 * 28)
                outputs = model(images)
                predicted = np.argmax(outputs, axis=1) if model_type == "custom" else torch.argmax(outputs, dim=1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)
        test_accuracy = test_correct / test_total

        end_time = time.time()
        results.append({
            "Hidden1": hidden1,
            "Hidden2": hidden2 if hidden2 else "N/A",
            "Test Accuracy": test_accuracy,
            "Training Time (s)": round(end_time - start_time, 2)
        })
    return results

hidden_size1_variants = [8, 16, 64, 1024]
hidden_size2_variants = [8, 16, 64, 1024]
remove_layer_config = [(256, None)]

configurations = [(h1, 128) for h1 in hidden_size1_variants] + \
                 [(256, h2) for h2 in hidden_size2_variants] + \
                 remove_layer_config

if __name__ == '__main__':
    dataset = FashionMNIST(batch_size=64, val_perc=0.2)
    pytorch_results = run_experiment(configurations, dataset, model_type="pytorch", lr=1e-2, epochs=20)
    custom_results = run_experiment(configurations, dataset, model_type="custom", lr=1e-2, epochs=20)

    import pandas as pd
    results_df = pd.DataFrame(pytorch_results + custom_results)
    results_df["Model"] = ["PyTorch"] * len(pytorch_results) + ["Custom"] * len(custom_results)
    print(results_df)
