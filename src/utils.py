import torch


def to_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")