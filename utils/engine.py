import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.mnet import MainNet
from models.afnet import AFNet
from models.hydraplus import HydraPlusNet
from typing import Iterable
from tqdm.auto import tqdm
import mlflow
import mlflow.pytorch as mp

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
                afnet_path: Iterable, **kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume = kwargs['resume']
    if model_name == "MainNet":
        net = MainNet()
        if not resume:
            net.apply(weight_init)
    elif "AF" in model_name:
        net = AFNet(af_name=model_name)
        if not resume:
            net.main_net.load_state_dict(torch.load(mnet_path))
        for param in net.main_net.parameters():
            param.requires_grad = False
    elif model_name == "HP":
        net = HydraPlusNet()
        if not resume:
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
    model.train()
    total_loss = []
    for imgs, targets, file_names in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    total_loss = np.mean(total_loss)
    return total_loss

@torch.inference_mode()
def eval_one_epoch(model, loader, loss_fn, **kwargs):
    model.eval()
    device = next(model.parameters()).device
    total_loss = []
    for imgs, targets, file_names in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        total_loss.append(loss.item())
    total_loss = np.mean(total_loss)
    return total_loss

def train_model(model_name: str, loaders, loss_fn, epochs: int, **kwargs):
    log_params=kwargs["log_params"]
    resume = kwargs["resume"]
    experiment_name = model_name
    model = define_model(model_name=model_name, 
                        mnet_path=kwargs['mnet_path'], 
                        afnet_path=kwargs["afnet_path"],
                        resume=resume)
    device = next(model.parameters()).device
    mGPUs = kwargs["mGPUs"]
    checkpoint = kwargs["checkpoint"]
    start = 1
    if resume:
        model.load_state_dict(torch.load(checkpoint))
        numeric_filter = filter(str.isdigit, checkpoint)
        numeric_string = "".join(numeric_filter)
        start = int(numeric_string) + 1
    
    if mGPUs:
        model = nn.DataParallel(model)
    model.train()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'], momentum=0.9)
    loss_fn = loss_fn.to(device)

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
        experiment_id = current_experiment['experiment_id']
    with mlflow.start_run(experiment_id=experiment_id):
        for epoch in tqdm(range(start, epochs + 1)):
            train_loss = train_one_epoch(model=model, loader=loaders["train"], optimizer=optimizer, loss_fn=loss_fn)
            val_loss = eval_one_epoch(model=model, loader=loaders["val"], loss_fn=loss_fn)
            print(f"Epoch: {epoch:4} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss})

            if mGPUs:
                checkpoint_save(model_name=model_name, state_dict=model.module.state_dict(), epoch=epoch)
            else:
                checkpoint_save(model_name=model_name, state_dict=model.state_dict(), epoch=epoch)
            
            if epoch % 20 == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.95
        mlflow.log_params(log_params)
        mlflow.end_run()
        del model


def predict(data_input, model_name, img_name, model, att_mode):
    if att_mode == "no_att":
        outputs = model(data_input)
    else:
        if model_name == "HP":
            att1, att2, att3, outputs = model(data_input)
            outputs_dict = {
                "file_name": img_name[0],
                "AF1": att1[0].detach().numpy(),
                "AF2": att2[0].detach().numpy(),
                "AF3": att3[0].detach().numpy()
            }
        elif "AF" in model_name:
            outputs, att = model(data_input)
            outputs_dict = {
                "file_name": img_name[0],
                model_name: att[0].detach().numpy()
            }
        else:
            raise ValueError
            
        if att_mode == "pkl_save":
            pickle.dump(
                outputs_dict,
                open(f"results/att_output_{model_name}.pkl", "ab")
            )
        else:
            pass
    return outputs

def test_model(model_name, loader, att_mode, weight_path, **kwargs):
    classes = pickle.load(open("assets/classes.pkl", "rb"))
    if att_mode == "no_att":
        if "AF" in model_name:
            model = AFNet(af_name=model_name)
        elif model_name == "HP":
            model = HydraPlusNet()
        elif model_name == "MainNet":
            model = MainNet()
        else:
            raise ValueError
    else:
        if "AF" in model_name:
            model = AFNet(att_out=True, af_name=model_name)
        elif model_name == "HP":
            model = HydraPlusNet(att_out=True)
        else:
            raise ValueError
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    print("[INFO] Load Parameter")
    model.eval()

    data_iter = iter(loader)
    count = 0
    TP = [0.0] * 26
    P  = [0.0] * 26
    TN = [0.0] * 26
    N  = [0.0] * 26

    acc = 0.0
    precision = 0.0
    recall = 0.0

    if att_mode == "pkl_save":
        pkl_file = f"results/att_output_{model_name}.pkl"
        if os.path.exists(pkl_file):
            os.remove(pkl_file)
    
    while count < len(loader):
        imgs, targets, file_names = data_iter.next()
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = predict(
            data_input=imgs,
            model_name=model_name,
            img_name=file_names,
            model=model,
            att_mode=att_mode
        )

        Yandf = 0.1
        Yorf = 0.1
        Y = 0.1
        f = 0.1
        i = 0

        for item in outputs[0]:
            if item.data.item() > 0:
                f = f+1
                Yorf += 1
                if targets[0][i].data.item() == 1:
                    TP[i] += 1
                    P[i] += 1
                    Y += 1
                    Yandf += 1
                else:
                    N[i] += 1
            else:
                if targets[0][i].data.item() == 0:
                    TN[i] += 1
                    N[i] += 1
                else:
                    P[i] += 1
                    Yorf += 1
                    Y += 1
            i += 1
        acc += (Yandf / Yorf)
        precision += (Yandf / f)
        recall += (Yandf / Y)
        if count % 1 == 0:
            print(f"Test on {count}-th image.")
        count += 1
    
    accuracy = 0
    print(f"TP: {TP} | TN: {TN} | P: {P} | N: {N}")
    for c in range(26):
        metric = (TP[c]/P[c] + TN[c]/N[c]) / 2
        print(f"{classes[c]}: {metric:.3f}")
        accuracy += (TP[c] / P[c] + TN[c] / N[c])
    mean_acc = accuracy / 52

    print(f"Path: {weight_path} | mA: {mean_acc:.3f}")

    acc /= 10000
    precision /= 10000
    recall /= 10000
    f1 = 2 * precision * recall / (precision + recall)

    print(f"Acc: {acc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")