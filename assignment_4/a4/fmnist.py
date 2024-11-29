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

### CODE FOR Test Performance STARTS HERE###

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

### CODE FOR Test Performance ENDS HERE###


### CODE FOR Validation Accuracy STARTS HERE###
# np.random.seed(33)

# class FashionMNIST:
#     def __init__(self, batch_size, val_perc: float = 0.2):
#         self.data_seed = 6666
#         np.random.seed(self.data_seed)
#         torch.manual_seed(self.data_seed)
#         torch.cuda.manual_seed(self.data_seed)
#         self.generator = torch.Generator().manual_seed(self.data_seed)

#         self.batch_size = batch_size
#         self.val_perc = val_perc
#         self.load_data()

#     def load_data(self):
#         transform = transforms.Compose([
#             transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
#         ])

#         self.train_dataset = torchvision.datasets.FashionMNIST(
#             root='./data', train=True, download=True, transform=transform
#         )
#         if self.val_perc is not None and self.val_perc > 0:
#             self.train_dataset, self.val_dataset = torch.utils.data.random_split(
#                 self.train_dataset,
#                 [
#                     int((1 - self.val_perc) * len(self.train_dataset)),
#                     int(self.val_perc * len(self.train_dataset))
#                 ],
#                 generator=self.generator
#             )
#             self.val_loader = torch.utils.data.DataLoader(
#                 self.val_dataset, batch_size=self.batch_size, shuffle=False,
#                 num_workers=4, pin_memory=True,
#                 generator=self.generator
#             )
            
#         self.test_dataset = torchvision.datasets.FashionMNIST(
#             root='./data', train=False, download=True, transform=transform
#         )
#         self.train_loader = torch.utils.data.DataLoader(
#             self.train_dataset, batch_size=self.batch_size, shuffle=True,
#             num_workers=4, pin_memory=True,
#             generator=self.generator
#         )
#         self.test_loader = torch.utils.data.DataLoader(
#             self.test_dataset, batch_size=self.batch_size, shuffle=False,
#             num_workers=4, pin_memory=True,
#             generator=self.generator
#         )
        

# class PytorchMLPFashionMNIST(nn.Module):
#     def __init__(
#         self,
#         input_size=28*28,
#         hidden_size1=256,
#         hidden_size2=128,
#         output_size=10
#     ):
#         super(PytorchMLPFashionMNIST, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_size, hidden_size1),
#             nn.ReLU(),
#             nn.Linear(hidden_size1, hidden_size2),
#             nn.ReLU(),
#             nn.Linear(hidden_size2, output_size),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#         return self.layers(x)

#     def train(
#         self,
#         data: FashionMNIST,
#         criterion: nn.Module=nn.CrossEntropyLoss(),
#         lr: float=0.001,
#     ):
#         self.optimizer = self.optimizer \
#             if hasattr(self, 'optimizer') else \
#                optim.SGD(self.parameters(), lr=lr)

#         running_loss = 0.0

#         for images, labels in data.train_loader:
#             self.optimizer.zero_grad()
#             outputs = self(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             self.optimizer.step()
#             running_loss += loss.item()
#         train_loss = running_loss / len(data.train_loader)

#         if data.val_perc is not None and data.val_perc > 0:
#             correct = 0
#             total = 0
#             with torch.no_grad():
#                 for images, labels in data.val_loader:
#                     outputs = self(images)
#                     predicted = torch.argmax(outputs, axis=1)
#                     total += labels.size(0)
#                     correct += (predicted == labels).sum().item()
#             val_accuracy = correct / total

#         return train_loss, val_accuracy

#     def test(self, data: FashionMNIST):
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in data.test_loader:
#                 outputs = self(images)
#                 predicted = torch.argmax(outputs, axis=1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         return correct/total


# class CustomMLPFashionMNIST(Layer):
#     def __init__(
#         self,
#         input_size=28*28,
#         hidden_size1=256,
#         hidden_size2=128,
#         output_size=10
#     ):
#         super(CustomMLPFashionMNIST, self).__init__()
#         self.layers = MLP([
#             Dense(input_size, hidden_size1),
#             ReLULayer(),
#             Dense(hidden_size1, hidden_size2),
#             ReLULayer(),
#             Dense(hidden_size2, output_size),
#             SoftmaxLayer()
#         ])

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         return self.layers(x)

#     def backward(self, grad):
#         return self.layers.backward(grad)

#     def update(self, lr):
#         self.layers.update(lr)

#     def train(
#         self,
#         data: FashionMNIST,
#         criterion: Layer=CrossEntropyLossLayer(),
#         lr: float=0.001,
#     ):
#         running_loss = 0.0

#         for images, labels in data.train_loader:
#             outputs = self(images.view(images.size(0), -1))
#             loss = criterion(outputs, labels.numpy())
#             loss_grad = criterion.backward(loss)
#             self.backward(loss_grad)
#             self.update(lr)
#             running_loss += loss.item()

#         train_loss = running_loss / len(data.train_loader)

#         if data.val_perc is not None and data.val_perc > 0:
#             correct = 0
#             total = 0
#             for images, labels in data.val_loader:
#                 outputs = self(images.view(images.size(0), -1))
#                 predicted = np.argmax(outputs, axis=1)
#                 total += labels.size(0)
#                 correct += (predicted == labels.numpy()).sum().item()
#             val_accuracy = correct / total

#         return train_loss, val_accuracy

#     def test(self, data: FashionMNIST):
#         correct = 0
#         total = 0
#         for images, labels in data.test_loader:
#             outputs = self(images.view(images.size(0), -1))
#             predicted = np.argmax(outputs, axis=1)
#             total += labels.size(0)
#             correct += (predicted == labels.numpy()).sum().item()

#         return correct/total


# def plot_combined(losses: dict, accuracies: dict, title='Comparison of Models'):
#     epochs = len(next(iter(losses.values())))
#     fig, axes = plt.subplots(1, 2, figsize=(15, 5))

#     for model, train_loss in losses.items():
#         axes[0].plot(np.arange(1, epochs+1), train_loss, label=model.upper())
#     axes[0].set_title('Training Loss')
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylabel('Loss')
#     axes[0].legend()

#     for model, val_acc in accuracies.items():
#         axes[1].plot(np.arange(1, epochs+1), val_acc, label=model.upper())
#     axes[1].set_title('Validation Accuracy')
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('Accuracy')
#     axes[1].legend()

#     plt.suptitle(title)
#     plt.show()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--models', nargs='+', default=['pytorch', 'custom'])
#     parser.add_argument('--lr', type=float, default=1e-2)
#     parser.add_argument('--epochs', type=int, default=20)
#     parser.add_argument('--batch-size', type=int, default=64)
#     parser.add_argument('--val-perc', type=float, default=0.2)
#     parser.add_argument('--plot', action='store_true', default=False)

#     args = parser.parse_args()

#     dataset = FashionMNIST(batch_size=args.batch_size, val_perc=args.val_perc)

#     model_losses = {}
#     model_accuracies = {}

#     for mod in args.models:
#         model = PytorchMLPFashionMNIST() if mod == 'pytorch' else CustomMLPFashionMNIST()

#         train_losses = []
#         val_accuracies = []

#         for epoch in tqdm(range(args.epochs), desc=f'Training {mod}'):
#             train_loss, val_acc = model.train(dataset, lr=args.lr)
#             train_losses.append(train_loss)
#             val_accuracies.append(val_acc)

#         model_losses[mod] = train_losses
#         model_accuracies[mod] = val_accuracies

#         test_acc = model.test(dataset)
#         print(f'[{mod.upper()}] Test accuracy: {test_acc:.2%}')

#     if args.plot:
#         plot_combined(model_losses, model_accuracies)
### CODE FOR Validation Accuracy ENDS HERE###