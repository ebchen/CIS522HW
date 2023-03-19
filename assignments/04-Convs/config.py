from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip


class CONFIG:
    batch_size = 32
    num_epochs = 3

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.AdamW(model.parameters(), lr=1e-3)

    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
    ])
