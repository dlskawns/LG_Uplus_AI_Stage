from model import NeuMF
from preprocessing import Data_Loader, PreProcess
from train import cfg, make_batchdata, update_avg, train_epoch
from test import valid_epoch
from collections import defaultdict
import torch

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os, random

from scipy import sparse
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import pickle
import os 

def main():
    preprocess = PreProcess()
    val_data = preprocess.valid
    if 'train_data.pkl' not in os.listdir('./data'):
        train_data = preprocess.make_UIdataset(cfg.neg_ratio)
        with open('./data/train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)
    with open('./data/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)

    u_feats = preprocess.user_sex_views_feat()
    i_feats = preprocess.item_run_feat()
    search_buy_dicts = preprocess.search_buy_dicts()

    # Model 선언
    model = NeuMF(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.reg_lambda)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    # training epoches
    total_logs = defaultdict(list)
    best_scores = 0

    for epoch in range(cfg.epochs+1):
        cfg.epoch = epoch
        train_results = train_epoch(cfg, train_data, model, optimizer, criterion)
        # cfg.check_epoch 번의 epoch 마다 성능 확인 
        if epoch % cfg.check_epoch == 0: 
            valid_results, _ = valid_epoch(cfg, u_feats, i_feats, search_buy_dicts, model, val_data)
            logs = {
                'Train Loss': train_results['losses'],
                f'Valid Recall@{cfg.top_k}': valid_results['recall'],
                f'Valid NDCG@{cfg.top_k}': valid_results['ndcg'],
                'Valid Coverage': valid_results['coverage'],
                'Valid Score': valid_results['score'],
                }

            # 검증 성능 확인 
            for key, value in logs.items():
                total_logs[key].append(value)

            if epoch == 0:
                print("Epoch", end=",")
                
                print(",".join(logs.keys()))

            print(f"{epoch:02d}  ", end="")
            print("  ".join([f"{v:0.6f}" for v in logs.values()]))

            # 가장 성능이 좋은 가중치 파일을 저장 
            if best_scores <= valid_results['score']: 
                best_scores = valid_results['score']
                torch.save(model.state_dict(), os.path.join('./save', 'model(best_scores).pth'))
    
if __name__ == '__main__':
    
    main()