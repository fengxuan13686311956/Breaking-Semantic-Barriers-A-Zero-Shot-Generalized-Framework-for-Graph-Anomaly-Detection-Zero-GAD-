import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import build_model


from utils import *

def pretrain_mae(args , adj_train , features_train,ano_label_train):

    # 打印预训练信息
    print("pretraining start")

    #model定义
    model = build_model(args)
    model.to(args.device)
    optimizer = create_optimizer(args.optimizer_mae, model, args.lr_mae, args.weight_decay_mae)
    print("model built")
    #model训练
    print("model pretrain:")
    model = train_mae(model,features_train,adj_train,ano_label_train, optimizer, args)
    model = model.cpu()
    # model = model.to(args.device)
    model.eval()
    print("model pretrained")
    return model

def train_mae(model, features,adjs, ano_labels,optimizer, args):
    model.train()
    max_epoch=args.max_epoch_mae
    device=args.device
    epoch_iter = tqdm(range(max_epoch))
    a=args.beta
    for epoch in epoch_iter:
        for feat,adj,ano_label in zip(features,adjs,ano_labels):

            recon,loss_mid= model(feat,adj)
            loss_gen=model.criterion(feat,recon)
            loss=loss_gen+a*loss_mid
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {loss:.4f}")

    return model
