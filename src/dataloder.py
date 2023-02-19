import numpy as np

import tqdm as tqdm
import torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.dataset.mnist import MNIST





class Dataloader:
    np.random.seed(42)

    def get_train_loader(dataset=MNIST, batch_size=64):
        train_set = MNIST(root="./../datasets", train=True, download=True, transform=ToTensor)

        return DataLoader(train_set, shuffle=True,batch_size=batch_size)
    

    def get_test_loader(dataset=MNIST, batch_size=64):
        test_set = MNIST(root="./../datasets", train=False, download=True, transform=ToTensor)

        return DataLoader(test_set, shuffle=False,batch_size=batch_size)
