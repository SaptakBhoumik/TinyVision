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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def accuracy_test(model, criterion, dataloader, device, num_classes=10):
    model.eval()
    total_loss = []
    confusion = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss.append(loss.item())

            predicted = outputs.argmax(dim=1)

            for c in range(num_classes):
                for p in range(num_classes):
                    confusion[c, p] += ((labels == c) & (predicted == p)).sum().item()

    accuracy = np.trace(confusion) / np.sum(confusion)

    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized_confusion = confusion / row_sums

    return np.mean(total_loss), accuracy, normalized_confusion


def load_model_for_inference(model,model_path):
    checkpoint = torch.load(model_path, map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model = torch.compile(model, mode="default", backend="inductor")
    return model

def train_f32_loop(
    model,
    optimizer,
    criterion,
    train_dataset,
    val_dataset,
    num_epochs,
    checkpoint_path,
    batch_size=128
):
    curr_epoch = 0

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = 2, pin_memory=True)
    testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    train_loss_array = []
    test_loss_array = []

    train_accuracy_array = []
    test_accuracy_array = []

    train_heatmap = []
    test_heatmap = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        curr_epoch = checkpoint.get("epoch", 0)

        train_loss_array = checkpoint.get("train_loss_array", [])
        train_accuracy_array = checkpoint.get("train_accuracy_array", [])

        test_accuracy_array = checkpoint.get("test_accuracy_array", [])
        test_loss_array = checkpoint.get("test_loss_array", [])

        train_heatmap = checkpoint.get("train_heatmap", [])
        test_heatmap = checkpoint.get("test_heatmap", [])
    


    print("Starting")
    for i in range(0, min(curr_epoch,num_epochs)):
        print(f"Epoch {i+1}/{num_epochs}: Train loss = {train_loss_array[i]:.4f}, Train accuracy = {train_accuracy_array[i]:.4f}, "
              f"Test loss = {test_loss_array[i]:.4f}, Test accuracy = {test_accuracy_array[i]:.4f}")
        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 2, 1)
        # sns.heatmap(train_heatmap[i], annot=False, fmt=".2f", cmap="Blues", cbar=False)
        # plt.title(f"Train Heatmap Epoch {i+1}")
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')    
        # plt.subplot(1, 2, 2)
        # sns.heatmap(test_heatmap[i], annot=False, fmt=".2f", cmap="Blues", cbar=False)
        # plt.title(f"Test Heatmap Epoch {i+1}")  
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.tight_layout()
        # plt.show()
        # plt.close()

    while curr_epoch < num_epochs:
        lr = 0
        if curr_epoch <= 30:
            lr = 0.01
        elif curr_epoch <= 50:
            lr = 0.001
        else:
            lr = 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        curr_epoch += 1
        print(f"Epoch {curr_epoch}/{num_epochs}: ", end='')
        
        loss,acc,heatmap0 = accuracy_test(model, criterion, trainloader, device)
        train_loss_array.append(loss)
        train_accuracy_array.append(acc)
        train_heatmap.append(heatmap0)
        print(f"Train loss = {train_loss_array[-1]:.4f}, Train accuracy = {train_accuracy_array[-1]:.4f}, ", end='')

        loss, acc,heatmap1 = accuracy_test(model, criterion, testloader, device)
        test_loss_array.append(loss)
        test_accuracy_array.append(acc)
        test_heatmap.append(heatmap1)
        print(f"Test loss = {test_loss_array[-1]:.4f}, Test accuracy = {test_accuracy_array[-1]:.4f}")
        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 2, 1)
        # sns.heatmap(train_heatmap[-1], annot=False, fmt=".2f", cmap="Blues", cbar=False)
        # plt.title(f"Train Heatmap Epoch {curr_epoch}")
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.subplot(1, 2, 2)
        # sns.heatmap(test_heatmap[-1], annot=False, fmt=".2f", cmap="Blues", cbar=False)
        # plt.title(f"Test Heatmap Epoch {curr_epoch}")
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.tight_layout()
        # plt.show()
        # plt.close()

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

        
    print("Training complete.")
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(train_heatmap[-1], annot=False, fmt=".2f", cmap="Blues", cbar=False)
    plt.title(f"Train Heatmap Epoch {curr_epoch}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.subplot(1, 2, 2)
    sns.heatmap(test_heatmap[-1], annot=False, fmt=".2f", cmap="Blues", cbar=False)
    plt.title(f"Test Heatmap Epoch {curr_epoch}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))

    epochs = range(1, len(train_loss_array) + 1)

    plt.figure(figsize=(12, 5))

    # ---------- Subplot 1 : Loss ----------
    plt.subplot(1, 2, 1)

    plt.plot(epochs, train_loss_array, label="Train Loss")
    plt.plot(epochs, test_loss_array, label="Test Loss")

    plt.axvline(32, linestyle='--')
    plt.axvline(52, linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")

    # plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    plt.legend()


    # ---------- Subplot 2 : Accuracy ----------
    plt.subplot(1, 2, 2)

    plt.plot(epochs, train_accuracy_array, label="Train Accuracy")
    plt.plot(epochs, test_accuracy_array, label="Test Accuracy")

    plt.axvline(32, linestyle='--')
    plt.axvline(52, linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")

    # plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    plt.legend()


    plt.tight_layout()
    plt.show()
    plt.close()


def he_initialization(model):
    # Applying He initialization (Kaiming Normal) to Conv2d and Linear layers
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(layer.weight, a = 0.25 , mode='fan_in', nonlinearity='leaky_relu')
            if layer.bias is not None:
                init.zeros_(layer.bias)
def describe(model,input_shape=(1, 3, 32, 32)):
    macs, params = get_model_complexity_info(model, input_shape[1::], as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f"MACs: {macs}, Params: {params}")
    print(summary(model, input_size=input_shape, device="cpu"))
    
def trainf32(model, path, criterion, train_dataset, val_dataset, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # <-- move this up BEFORE collecting params

    decay = []
    no_decay = []

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            decay.append(module.weight)
            if module.bias is not None:
                no_decay.append(module.bias)
        else:
            for name, param in module.named_parameters(recurse=False):
                no_decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.0001},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=0.01,
    )

    train_f32_loop(
        model,
        optimizer,
        criterion,
        train_dataset,
        val_dataset,
        60,
        path,
        batch_size=batch_size
    )


def get_dataset():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Lambda(img_layernorm),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(img_layernorm),
    ])

    train = datasets.CIFAR10(
        "../../../dataset/Cifar10", train=True, transform=train_transform, download=True
    )
    test = datasets.CIFAR10(
        "../../../dataset/Cifar10", train=False, transform=test_transform, download=True
    )

    return train, test