"""Lab 3: data loaders and the three models (MLP, CNN, patch attention)."""
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_ROOT = r"F:\datasets"
BATCH_SIZE = 128
SEED = 0


# ---------- Task 1: Fashion-MNIST ----------

def get_loaders(root=DATA_ROOT, batch_size=BATCH_SIZE):
    """Official 60k train -> 55k train + 5k val; official 10k test."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_full = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    # fixed generator -> the same split on every run
    train, val = random_split(train_full, [55_000, 5_000],
                              generator=torch.Generator().manual_seed(SEED))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )


# ---------- Task 2: models ----------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class PatchAttention(nn.Module):
    """28x28 -> 16 patches of 7x7 -> Linear(49, 64) -> 2 encoder layers -> mean pool -> Linear(64, 10).

    Exactly as in the manual: no positional embedding. This makes the model
    blind to patch order - worth discussing in the report.
    """

    def __init__(self, d_model=64, nhead=4, dim_ff=128, num_layers=2, patch=7):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(patch * patch, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):                                       # B,1,28,28
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)                   # B,1,4,4,7,7
        x = x.reshape(x.size(0), -1, p * p)                     # B,16,49
        x = self.encoder(self.proj(x))                          # B,16,64
        return self.head(x.mean(dim=1))                         # B,10


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": PatchAttention}
