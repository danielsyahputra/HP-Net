import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.mnet import MainNet
from models.afnet import AFNet
from models.hydraplus import HydraPlusNet
from typing import Dict, Iterable
from tqdm.auto import tqdm

def weight_init(model) -> None:
    if isinstance(model, nn.Conv2d):
        nn.init.xavier_normal_(model.weight.data)

def checkpoint_save(model_name: str, state_dict, epoch) -> None:
    root_dir = f"checkpoint/{model_name}"
    os.makedirs(root_dir, exist_ok=True)
    save_path = f"{root_dir}/{model_name}_epoch_{epoch}"
    torch.save(state_dict, save_path)

def define_model(model_name: str, 
                mnet_path: str,
                afnet_path: Iterable):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "MainNet":
        net = MainNet()
        net.apply(weight_init)
    elif "AF" in model_name:
        net = AFNet(af_name=model_name)
        net.main_net.load_state_dict(torch.load(mnet_path))
        for param in net.main_net.parameters():
            param.requires_grad = False
    elif model_name == "HP":
        net = HydraPlusNet()
        net.main_net.load_state_dict(torch.load(mnet_path))
        net.af1.load_state_dict(torch.load(afnet_path[0]))
        net.af2.load_state_dict(torch.load(afnet_path[1]))
        net.af3.load_state_dict(torch.load(afnet_path[2]))
        
        # Freeze other network
        for param in net.main_net.parameters():
            param.requires_grad = False
        for param in net.af1.parameters():
            param.requires_grad = False
        for param in net.af2.parameters():
            param.requires_grad = False
        for param in net.af3.parameters():
            param.requires_grad = False
    else:
        raise ValueError
    return net.to(device)

def train_one_epoch(model,
                    loader,
                    optimizer,
                    loss_fn):

    device = next(model.parameters()).device

    total_loss = []
    for imgs, targets in loader:
        targets = torch.Tensor(targets).unsqueeze(0)
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    total_loss = np.mean(total_loss)
    return total_loss


def train_model(model_name: str, loaders: Dict, loss_fn, epochs: int, **kwargs):
    model = define_model(model_name=model_name, mnet_path=kwargs['mnet_path'], afnet_path=kwargs["afnet_path"])
    device = next(model.parameters()).device
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'], momentum=0.9)
    loss_fn = loss_fn.to(device)
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train_one_epoch(model=model, loader=loaders["train"], optimizer=optimizer, loss_fn=loss_fn)
        print(f"Epoch: {epoch:4} | Train Loss: {train_loss:.3f}")

        if epoch % 5 == 0:
            checkpoint_save(model_name=model_name, state_dict=model.state_dict(), epoch=epoch)
            for param_group in optimizer.param_groups():
                param_group["lr"] *= 0.95
    del model