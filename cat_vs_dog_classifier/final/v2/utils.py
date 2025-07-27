import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.init as init
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchinfo import summary
import seaborn as sns
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
import os

class SplitLoader:
    def __init__(self, data_path,files, batch_size):
        cat_chunks = []
        dog_chunks = []
        for file in files["cat"]:
            cat_chunks.append(os.path.join(data_path, 'cat', "lr", file))
            cat_chunks.append(os.path.join(data_path, 'cat', "original", file))
        for file in files["dog"]:
            dog_chunks.append(os.path.join(data_path, 'dog', "lr", file))
            dog_chunks.append(os.path.join(data_path, 'dog', "original", file))
        cat_chunks = sorted(cat_chunks)
        dog_chunks = sorted(dog_chunks)
            
        # self.data_path = data_path
        np.random.seed(42)  # For reproducibility
        self.cat_chunks = np.random.permutation(cat_chunks)
        self.dog_chunks = np.random.permutation(dog_chunks)
        self.batch_size = batch_size

        self.dataloader = None

    def _load_chunk(self,idx):
        x_cat_tensor = np.load(self.cat_chunks[idx])
        x_dog_tensor = np.load(self.dog_chunks[idx])

        x_tensor = np.concatenate((x_cat_tensor, x_dog_tensor), axis=0)
        y_tensor = np.concatenate((np.zeros(x_cat_tensor.shape[0]), np.ones(x_dog_tensor.shape[0])), axis=0)

        self.dataloader = DataLoader(
            TensorDataset(torch.tensor(x_tensor, dtype=torch.float32), torch.tensor(y_tensor, dtype=torch.long)),
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def __iter__(self):
        for idx in range(len(self.cat_chunks)):
            self._load_chunk(idx)
            for batch in self.dataloader:
                yield batch

    def reshuffle(self):
        self.cat_chunks = np.random.permutation(self.cat_chunks)
        self.dog_chunks = np.random.permutation(self.dog_chunks)
        self._load_chunk(0)

def accuracy_test(model, criterion, dataloader, device):
    model.eval()
    total_loss = []
    heatmap = np.zeros((2, 2))  # For confusion matrix
    

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss.append(loss.item())

            predicted = outputs.argmax(dim=1)

            heatmap[0, 0] += ((predicted == 0) & (labels == 0)).sum().item()
            heatmap[0, 1] += ((predicted == 1) & (labels == 0)).sum().item()
            heatmap[1, 0] += ((predicted == 0) & (labels == 1)).sum().item()
            heatmap[1, 1] += ((predicted == 1) & (labels == 1)).sum().item()

    heatmap = heatmap.astype(float)
    heatmap /= heatmap.sum(axis=1, keepdims=True)  # Normalize by row to get proportions
    
    return np.mean(total_loss), np.mean(np.diag(heatmap)), heatmap

def load_model_for_inference(model,model_path):
    checkpoint = torch.load(model_path, map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.compile(model, mode="default", backend="aot_eager")
    return model

def train_loop(
    model,
    optimizer,
    criterion,
    train_dataset,
    val_dataset,
    num_epochs,
    checkpoint_path,
):
    curr_epoch = 0

    train_loss_array = []
    test_loss_array = []

    train_accuracy_array = []
    test_accuracy_array = []

    train_heatmap = []
    test_heatmap = []
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        curr_epoch = checkpoint.get("epoch", 0)

        train_loss_array = checkpoint.get("train_loss_array", [])
        train_accuracy_array = checkpoint.get("train_accuracy_array", [])

        test_accuracy_array = checkpoint.get("test_accuracy_array", [])
        test_loss_array = checkpoint.get("test_loss_array", [])

        train_heatmap = checkpoint.get("train_heatmap", [])
        test_heatmap = checkpoint.get("test_heatmap", [])
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.compile(model, mode="default", backend="aot_eager")

    print("Starting")
    for i in range(0, min(curr_epoch,num_epochs)):
        print(f"Epoch {i+1}/{num_epochs}: Train loss = {train_loss_array[i]:.4f}, Train accuracy = {train_accuracy_array[i]:.4f}, "
              f"Test loss = {test_loss_array[i]:.4f}, Test accuracy = {test_accuracy_array[i]:.4f}")
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        sns.heatmap(train_heatmap[i], annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=['Predicted Cat', 'Predicted Dog'],
                    yticklabels=['Actual Cat', 'Actual Dog'])
        plt.title(f"Train Heatmap Epoch {i+1}")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')    
        plt.subplot(1, 2, 2)
        sns.heatmap(test_heatmap[i], annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=['Predicted Cat', 'Predicted Dog'],
                    yticklabels=['Actual Cat', 'Actual Dog'])
        plt.title(f"Test Heatmap Epoch {i+1}")  
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        plt.close()

    while curr_epoch < num_epochs:
        lr = 0
        if curr_epoch <= 6:
            lr = 0.001
        elif curr_epoch <= 9:
            lr = 0.0001
        else:
            lr = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        for x, y in train_dataset:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        curr_epoch += 1
        print(f"Epoch {curr_epoch}/{num_epochs}: ", end='')
        
        loss,acc,heatmap0 = accuracy_test(model, criterion, train_dataset, device)
        train_loss_array.append(loss)
        train_accuracy_array.append(acc)
        train_heatmap.append(heatmap0)
        print(f"Train loss = {train_loss_array[-1]:.4f}, Train accuracy = {train_accuracy_array[-1]:.4f}, ", end='')

        loss, acc,heatmap1 = accuracy_test(model, criterion, val_dataset, device)
        test_loss_array.append(loss)
        test_accuracy_array.append(acc)
        test_heatmap.append(heatmap1)
        print(f"Test loss = {test_loss_array[-1]:.4f}, Test accuracy = {test_accuracy_array[-1]:.4f}")

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        sns.heatmap(train_heatmap[-1], annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=['Predicted Cat', 'Predicted Dog'],
                    yticklabels=['Actual Cat', 'Actual Dog'])
        plt.title(f"Train Heatmap Epoch {curr_epoch}")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.subplot(1, 2, 2)
        sns.heatmap(test_heatmap[-1], annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=['Predicted Cat', 'Predicted Dog'],
                    yticklabels=['Actual Cat', 'Actual Dog'])
        plt.title(f"Test Heatmap Epoch {curr_epoch}")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        plt.close()

        torch.save({
            "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            "epoch": curr_epoch ,

            "train_accuracy_array": train_accuracy_array,
            "test_accuracy_array": test_accuracy_array,

            "train_loss_array": train_loss_array,
            "test_loss_array": test_loss_array,

            "train_heatmap": train_heatmap,
            "test_heatmap": test_heatmap
        }, checkpoint_path)


        train_dataset.reshuffle()
        
    print("Training complete.")
    #Plot the training and test loss and accuracy
    plt.figure(figsize=(12, 6))

    # Subplot 1 - Loss
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range(1, 1 + len(train_loss_array)), y=train_loss_array, label='Train Loss')
    sns.lineplot(x=range(1, 1 + len(test_loss_array)), y=test_loss_array, label='Test Loss')

    plt.axvline(x=6, color='gray', linestyle='--')
    plt.axvline(x=9, color='gray', linestyle='--')
    # plt.axvline(x=10, color='gray', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.xticks(range(1, 1 + len(train_loss_array)))  # Show every integer tick
    plt.legend()

    # Subplot 2 - Accuracy
    plt.subplot(1, 2, 2)
    sns.lineplot(x=range(1, 1 + len(train_accuracy_array)), y=train_accuracy_array, label='Train Accuracy')
    sns.lineplot(x=range(1, 1 + len(test_accuracy_array)), y=test_accuracy_array, label='Test Accuracy')

    plt.axvline(x=6, color='gray', linestyle='--')
    plt.axvline(x=9, color='gray', linestyle='--')
    # plt.axvline(x=10, color='gray', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xticks(range(1, 1 + len(train_accuracy_array)))  # Show every integer tick
    plt.legend()

    plt.show()
    plt.close()


def he_initialization(model):
    # Applying He initialization (Kaiming Normal) to Conv2d and Linear layers
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(layer.weight, a = 0.25 , mode='fan_in', nonlinearity='leaky_relu')
            if layer.bias is not None:
                init.zeros_(layer.bias)
def describe(model,input_shape=(1, 10, 128, 128)):
    macs, params = get_model_complexity_info(model, input_shape[1::], as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f"MACs: {macs}, Params: {params}")
    print(summary(model, input_size=input_shape, device="cpu"))

def train(model,path,criterion,train_dataset,val_dataset,epoches=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay = 0.0001)
    train_loop(
        model,
        optimizer,
        criterion,
        train_dataset,
        val_dataset,
        epoches,
        path,
    )
