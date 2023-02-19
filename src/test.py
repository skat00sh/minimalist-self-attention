from tqdm import tqdm

import torch

from dataloder import Dataloader
from utils import to_device

from torch.nn import CrossEntropyLoss

def test(model):
    device = to_device()

    test_loader = Dataloader.get_test_loader()
    loss_metric = CrossEntropyLoss()

    with torch.no_grad():
        correct, total = 0,0
        test_loss = 0

        for batch in tqdm(test_loader, desc="Testing"):
            x,y = batch
            x,y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_metric(y_pred,y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_pred, dim=1)==y).detach().cpu().item()
            total += len(x)

            print(f"{test_loss = :.2f}")
            print(f"Test Accuracy : {correct / total *100:.2f}%")

            
