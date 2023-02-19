from tqdm import tqdm

import torch
from torch.optim  import Adam
from torch.nn import CrossEntropyLoss

# from config import basic_config
from dataloder import Dataloader

def train(model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")
    LR = 0.005
    EPOCHS = 5
    optimizer = Adam(model.parameters(),lr = LR)
    loss_metric = CrossEntropyLoss()
    train_loader = Dataloader.get_train_loader()


    for epoch in tqdm(range(EPOCHS),desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            x,y = batch
            x,y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_metric(y_pred,y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1} finished | Loss: {train_loss:.2f}")


            
             
