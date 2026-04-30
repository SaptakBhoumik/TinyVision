import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def round_no_zero(x):
    r = torch.round(x)
    return torch.where(r == 0, torch.sign(x), r)

def quant8_linear(W):
    # 1) per-neuron mean
    mu = W.mean(dim=1, keepdim=True)

    # 2) center weights
    Wc = W - mu

    # 3) 90th percentile per neuron
    n = Wc.size(1)
    k = int(n * 0.90)

    thresh, _ = Wc.abs().kthvalue(k, dim=1, keepdim=True)
    thresh = torch.clamp(thresh, min=1e-5) / 4  # NOTE: /4 for 8 levels

    # 4) quantize to {-4,-3,-2,-1,1,2,3,4}
    Wn = Wc / thresh
    Wq = round_no_zero(Wn)
    Wq = torch.clamp(Wq, -4, 4)

    # 5) restore mean (STE preserved)
    return W + (Wq * thresh + mu - W).detach()


def quant8_conv_per_kernel(W):
    # W shape [out, in, kH, kW]

    # 1) mean per kernel
    mu = W.mean(dim=(2, 3), keepdim=True)

    # 2) center
    Wc = W - mu

    # 3) 90th percentile per kernel
    flat = Wc.abs().reshape(W.size(0), W.size(1), -1)
    N = flat.size(2)
    k = int(N * 0.90)

    thresh, _ = flat.kthvalue(k, dim=2)
    thresh = thresh.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-5) / 4

    # 4) quantize to {-4,-3,-2,-1,1,2,3,4}
    Wn = Wc / thresh
    Wq = round_no_zero(Wn)
    Wq = torch.clamp(Wq, -4, 4)

    # 5) restore mean (STE)
    return W + (Wq * thresh + mu - W).detach()

class DiscreteLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # latent weight
        self.latent_weight = nn.Parameter(3 * torch.randn(out_features, in_features))

        # learned per-neuron scale 
        self.w_scale = nn.Parameter(torch.ones(out_features).reshape(-1, 1))

        # learned input scale
        self.input_scale = nn.Parameter(torch.ones(1))

        # bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    def optim_groups(self, weight_decay):
        decay = [self.latent_weight]
        no_decay = [self.w_scale, self.input_scale, self.bias]
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    
    def forward(self, x):
        W = quant8_linear(self.latent_weight) * self.w_scale
        return F.linear(x  * self.input_scale, W, self.bias)

class DiscreteConv2d(nn.Module):
    """
    Pentary activation {-2..2} (no learned activation scale)
    Weight quantization: W = a * W_int  per (out,in)
    User must quantize the FIRST LAYER input manually.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,padding_mode="zeros"):
        super().__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size,int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        # latent weights
        self.latent_weight = nn.Parameter(
            3 * torch.randn(
                out_channels,
                self.in_channels // groups,
                self.kernel_size[0],
                self.kernel_size[1])
        )

        # learned scale per (out,in)
        self.w_scale = nn.Parameter(
            torch.ones(out_channels, self.in_channels // groups)
        )
        # learned input scale
        self.input_scale = nn.Parameter(torch.ones(1, self.in_channels, 1, 1))

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _pad_args(self):
        ph, pw = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        return (pw, pw, ph, ph)

    def optim_groups(self, weight_decay):
        decay = [self.latent_weight]
        no_decay = [self.w_scale, self.input_scale, self.bias]
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    def forward(self, x):
        # scale input
        x = x * self.input_scale
        
        # ---------------------------
        # WEIGHT QUANTIZATION
        # ---------------------------
        W = quant8_conv_per_kernel(self.latent_weight) * self.w_scale.unsqueeze(-1).unsqueeze(-1)

        # ---------------------------
        # CONVOLUTION OP
        # ---------------------------
        if self.padding_mode != "zeros":
            x = F.pad(x, self._pad_args(), mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding

        z = F.conv2d(
            x, W, bias=self.bias,
            stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.groups
        )

        return z


class DiscreteBatchNorm(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False, track_running_stats=True)
        
    def forward(self, x):
        return self.bn(x)

class PACTQuant(nn.Module):
    def __init__(self, init_alpha=8.0, qmax=8):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.qmax = qmax
        
    def optim_groups(self, weight_decay):
        decay = []
        no_decay = [self.alpha]
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def forward(self, x):
        # PACT clipping
        x = torch.min(torch.relu(x), self.alpha)

        # quantize (STE)
        scale = (self.qmax-1) / self.alpha
        x_q = torch.round(x * scale) / scale

        # STE -> return quantized forward, real-value backward
        return x + (x_q - x).detach()



def quant7_act_per_channel(x):
    """
    x: [B, C, H, W]
    Quantize each channel to 7 levels {-3..3}
    Using same logic as quant7_conv_per_kernel but for activations.
    """
    # 1) mean per channel
    mu = x.mean(dim=[2,3], keepdim=True)    # [B,C,1,1]

    # 2) center
    xc = x - mu

    # 3) 90th percentile per channel
    B, C, H, W = x.shape
    flat = xc.abs().reshape(B, C, -1)       # [B,C,H*W]

    k = int((H*W) * 0.90)
    thresh,_ = flat.kthvalue(k, dim=2)      # [B,C]
    thresh = thresh.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-5) / 3  # [B,C,1,1]

    # 4) quantize
    xn = xc / thresh
    xq = torch.clamp(torch.round(xn), -3, 3)

    # 5) restore mean
    return x + (xq * thresh + mu - x).detach()


def quant8_act_per_channel(x):
    """
    x: [B, C, H, W]
    Quantize each channel to 7 levels {-3..3}
    Using same logic as quant7_conv_per_kernel but for activations.
    """
    # 1) 90th percentile per channel
    B, C, H, W = x.shape
    flat = x.reshape(B, C, -1)       # [B,C,H*W]

    k = int((H*W) * 0.90)
    thresh,_ = flat.kthvalue(k, dim=2)      # [B,C]
    thresh = thresh.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-5) / 7  # [B,C,1,1]

    # 2) quantize
    xn = x / thresh
    xq = torch.clamp(torch.round(xn), 0, 7)

    # 3) restore mean
    return x + (xq * thresh - x).detach()

class Quantize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return quant8_act_per_channel(x)

# def img_layernorm(x):
#     # x: [C,H,W]
#     mean = x.mean(dim=[1,2], keepdim=True)          # per-channel
#     std  = x.std(dim=[1,2], keepdim=True)
#     return (x - mean) / (std + 1e-6)

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
        "../../dataset/Cifar10", train=True, transform=train_transform, download=True
    )
    test = datasets.CIFAR10(
        "../../dataset/Cifar10", train=False, transform=test_transform, download=True
    )

    return train, test

def test(modelA, lr=1e-2, epochs=100, batch_size=64, patience=5, factor=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, testset = get_dataset()
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False)

    modelA = modelA.to(device)
    
    total = sum(p.numel() for p in modelA.parameters())
    grouped = sum(p.numel() for g in modelA.optim_groups(1e-4) for p in g["params"])

    print(f"Total parameters: {total}, Handled in optim_groups: {grouped}")
    assert total == grouped, "Some parameters are missing or duplicated"

    optimizer = optim.AdamW(modelA.optim_groups(weight_decay=1e-4), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("\n=============== Training Model ===============\n")

    best_loss = float("inf")
    epochs_since_improve = 0

    for ep in range(epochs):
        modelA.train()
        running_loss = 0
        correct = 0
        total = 0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = modelA(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_loss = running_loss / total
        avg_acc  = correct / total

        print(f"Epoch {ep+1:03d} | Loss={avg_loss:.4f} | Acc={avg_acc:.4f} | LR={optimizer.param_groups[0]['lr']:.5f}")

        # =========================
        #    LR DECAY IF STALLED
        # =========================
        if avg_loss < best_loss - 1e-4:       # improvement margin
            best_loss = avg_loss
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= patience:
            for g in optimizer.param_groups:
                g["lr"] *= factor              # reduce LR
            print(f" ↓ Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}\n")
            epochs_since_improve = 0           # reset counter

    # ---------------------------------------------------------
    # Final testing
    # ---------------------------------------------------------
    modelA.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = modelA(x)
            loss = criterion(out, y)

            test_loss += loss.item() * x.size(0)
            _, pred = torch.max(out, 1)
            test_correct += (pred == y).sum().item()
            test_total   += y.size(0)

    print(f"\nTest Loss={test_loss/test_total:.4f} | Test Acc={test_correct/test_total:.4f}\n")





pentary_modelA = nn.Sequential(
    Quantize(),# input quantization
    nn.BatchNorm2d(3),
    # -------- Block 1 --------
    DiscreteConv2d(3, 30, 3, padding=1,groups=3),
    DiscreteBatchNorm(30),
    nn.ReLU(),

    nn.Conv2d(30, 30, 1),   # pointwise conv
    nn.BatchNorm2d(30),
    PACTQuant(),

    DiscreteConv2d(30, 30, 3, padding=1,groups=30),
    DiscreteBatchNorm(30),
    nn.MaxPool2d(2),          # 32 → 16
    nn.ReLU(),

    nn.Conv2d(30, 30, 1),   # pointwise conv
    nn.BatchNorm2d(30),
    PACTQuant(),
    # nn.Dropout(0.2),

    # -------- Block 2 --------
    DiscreteConv2d(30, 60, 3, padding=1,groups=30),
    DiscreteBatchNorm(60),
    nn.ReLU(),

    nn.Conv2d(60, 60, 1),   # pointwise conv
    nn.BatchNorm2d(60),
    PACTQuant(),

    DiscreteConv2d(60, 60, 3, padding=1,groups=60),
    DiscreteBatchNorm(60),
    nn.MaxPool2d(2),          # 16 → 8
    nn.ReLU(),

    nn.Conv2d(60, 60, 1),   # pointwise conv
    nn.BatchNorm2d(60),
    PACTQuant(),
    # nn.Dropout(0.3),

    # -------- Block 3 --------
    DiscreteConv2d(60, 120, 3, padding=1, groups=60),
    DiscreteBatchNorm(120),
    nn.ReLU(),

    nn.Conv2d(120, 120, 1),   # pointwise conv
    nn.BatchNorm2d(120),
    PACTQuant(),

    DiscreteConv2d(120, 120, 3, padding=1, groups=120),
    DiscreteBatchNorm(120),
    nn.MaxPool2d(2),          # 8 → 4
    nn.ReLU(),

    nn.Conv2d(120, 120, 1),   # pointwise conv
    nn.BatchNorm2d(120),
    PACTQuant(),
    # nn.Dropout(0.4),

    # -------- Block 4 --------
    DiscreteConv2d(120, 60, 3, padding=1, groups=60),
    DiscreteBatchNorm(60),
    nn.ReLU(),

    nn.Conv2d(60, 60, 1),   # pointwise conv
    nn.BatchNorm2d(60),
    PACTQuant(),

    DiscreteConv2d(60, 60, 3, padding=1, groups=60),
    DiscreteBatchNorm(60),
    nn.MaxPool2d(2),          # 4 → 2
    nn.ReLU(),

    nn.Conv2d(60, 60, 1),   # pointwise conv
    nn.BatchNorm2d(60),
    PACTQuant(),
    # nn.Dropout(0.5),


    # -------- Head --------
    # nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    DiscreteLinear(240, 10)
)



regular_modelA = nn.Sequential(
    nn.BatchNorm2d(3),
    # -------- Block 1 --------
    nn.Conv2d(3, 30, 3, padding=1, groups=3),
    nn.BatchNorm2d(30),
    nn.ReLU(),

    nn.Conv2d(30, 30, 1),   # pointwise conv
    nn.BatchNorm2d(30),
    nn.ReLU(),

    nn.Conv2d(30, 30, 3, padding=1, groups=30),
    nn.BatchNorm2d(30),
    nn.MaxPool2d(2),          # 32 → 16
    nn.ReLU(),

    nn.Conv2d(30, 30, 1),   # pointwise conv
    nn.BatchNorm2d(30),
    nn.ReLU(),
    # nn.Dropout(0.2),

    # -------- Block 2 --------
    nn.Conv2d(30, 60, 3, padding=1,groups=30),
    nn.BatchNorm2d(60),
    nn.ReLU(),

    nn.Conv2d(60, 60, 1),   # pointwise conv
    nn.BatchNorm2d(60),
    nn.ReLU(),

    nn.Conv2d(60, 60, 3, padding=1,groups=60),
    nn.BatchNorm2d(60),
    nn.MaxPool2d(2),          # 16 → 8
    nn.ReLU(),

    nn.Conv2d(60, 60, 1),   # pointwise conv
    nn.BatchNorm2d(60),
    nn.ReLU(),
    # nn.Dropout(0.3),

    # -------- Block 3 --------
    nn.Conv2d(60, 120, 3, padding=1, groups=60),
    nn.BatchNorm2d(120),
    nn.ReLU(),

    nn.Conv2d(120, 120, 1),   # pointwise conv
    nn.BatchNorm2d(120),
    nn.ReLU(),

    nn.Conv2d(120, 120, 3, padding=1, groups=120),
    nn.BatchNorm2d(120),
    nn.MaxPool2d(2),          # 8 → 4
    nn.ReLU(),

    nn.Conv2d(120, 120, 1),   # pointwise conv
    nn.BatchNorm2d(120),
    nn.ReLU(),
    # nn.Dropout(0.4),

    # -------- Block 4 --------
    nn.Conv2d(120, 60, 3, padding=1, groups=60),
    nn.BatchNorm2d(60),
    nn.ReLU(),

    nn.Conv2d(60, 60, 1),   # pointwise conv
    nn.BatchNorm2d(60),
    nn.ReLU(),

    nn.Conv2d(60, 60, 3, padding=1, groups=60),
    nn.BatchNorm2d(60),
    nn.MaxPool2d(2),          # 4 → 2
    nn.ReLU(),

    nn.Conv2d(60, 60, 1),   # pointwise conv
    nn.BatchNorm2d(60),
    nn.ReLU(),
    # nn.Dropout(0.5),


    # -------- Head --------
    # nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(240, 10)
)
class ParentModule(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def optim_groups(self, weight_decay):
        groups = []
        handled = set()

        # 1️⃣ Ask smart child modules (skip self!)
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, "optim_groups"):
                for g in m.optim_groups(weight_decay):
                    groups.append(g)
                    for p in g["params"]:
                        handled.add(p)

        # 2️⃣ Fallback for everything else
        decay = []
        no_decay = []

        for module in self.modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad or param in handled:
                    continue

                # ---- BatchNorm: never decay ----
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    no_decay.append(param)
                    continue

                # ---- Bias of Linear / Conv: never decay ----
                if name == "bias" and isinstance(
                    module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
                ):
                    no_decay.append(param)
                    continue

                # ---- General rule ----
                if param.ndim >= 2:
                    decay.append(param)
                else:
                    no_decay.append(param)

        if decay:
            groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})

        return groups

    def forward(self, x):
        return self.net(x)


