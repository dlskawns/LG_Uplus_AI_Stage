import warnings
warnings.filterwarnings('ignore')

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


class cfg: 
    gpu_idx = 0
    device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")
    top_k = 25
    seed = 42
    neg_ratio = 100
    test_size = 0.1
    batch_size = 256
    emb_dim = 256
    layer_dim = 256
    dropout = 0.05
    epochs = 15
    learning_rate = 0.0025
    reg_lambda = 0
    check_epoch = 1
    n_continuous_feats = 4
    n_sex = 2
    n_search = 2
    

def make_batchdata(dataset, user_indices, batch_idx, batch_size):
    """ 배치 데이터로 변환 
    Args:
        user_indices : 전체 유저의 인덱스 정보 
            ex) array([ 3100,  1800, 30098, ...,  2177, 11749, 20962])
        batch_idx : 배치 인덱스 (몇번째 배치인지)
            ex) 0 
        batch_size : 배치 크기 
            ex) 256 
    Returns 
        batch_user_ids : 배치내의 유저 인덱스 정보 
            ex) [22194, 22194, 22194, 22194, 22194, ...]
        batch_item_ids : 배치내의 아이템 인덱스 정보 
            ex) [36, 407, 612, 801, 1404, ...]
        batch_feat0 : 배치내의 유저-아이템 인덱스 정보에 해당하는 feature0 정보 
            ex) [6, 6, 6, 6, 6, ...]
        batch_feat1 : 배치내의 유저-아이템 인덱스 정보에 해당하는 feature1 정보 
            ex) [4,  4,  4, 23,  4, ...]
        batch_labels : 배치내의 유저-아이템 인덱스 정보에 해당하는 label 정보 
            ex) [1.0, 1.0, 1.0, 1.0, 1.0, ...]
    """
    batch_user_indices = user_indices[batch_idx*batch_size : (batch_idx+1)*batch_size]
    batch_user_ids = []
    batch_item_ids = []
    batch_feat0 = []  # age
    batch_feat1 = []  # views
    batch_feat2 = []  # sex 
    batch_feat3 = []  # genre
    batch_feat4 = []  # runtime
    batch_feat5 = []  # search
    batch_feat6 = []  # payment
    # batch_feat7 = []  # keyword
    batch_labels = [] # label
    for user_id in batch_user_indices:
        item_ids = dataset[user_id][0]
        feat0 = dataset[user_id][1]
        feat1 = dataset[user_id][2]
        feat2 = dataset[user_id][3]
        feat3 = dataset[user_id][4]
        feat4 = dataset[user_id][5]
        feat5 = dataset[user_id][6]
        feat6 = dataset[user_id][7]                
        # feat7 = dataset[user_id][8] 
        labels = dataset[user_id][8]
        user_ids = np.full(len(item_ids), user_id)  # 모든 값을 user_id로 통일
        batch_user_ids.extend(user_ids.tolist())
        batch_item_ids.extend(item_ids.tolist())
        batch_feat0.extend(feat0.tolist())
        batch_feat1.extend(feat1.tolist())
        batch_feat2.extend(feat2.tolist())
        batch_feat3.extend(feat3.tolist())
        batch_feat4.extend(feat4.tolist())      
        batch_feat5.extend(feat5.tolist())   
        batch_feat6.extend(feat6.tolist())   
        # batch_feat7.extend(feat7)
        batch_labels.extend(labels.tolist())
    return batch_user_ids, batch_item_ids, batch_feat0, batch_feat1, batch_feat2, batch_feat3, batch_feat4,batch_feat5, batch_feat6, batch_labels # , batch_feat7

def update_avg(curr_avg, val, idx):
    """ 현재 epoch 까지의 평균 값을 계산 
    """
    return (curr_avg * idx + val) / (idx + 1)

def train_epoch(cfg, data, model, optimizer, criterion): 
    model.train()
    curr_loss_avg = 0.0

    user_indices = np.arange(cfg.n_users)
    np.random.RandomState(cfg.epoch).shuffle(user_indices)
    

    batch_num = int(len(user_indices) / cfg.batch_size) + 1
    bar = tqdm(range(batch_num), leave=False)
    for step, batch_idx in enumerate(bar):
        user_ids, item_ids, feat0, feat1,feat2, feat3, feat4,feat5, feat6, labels = make_batchdata(data, user_indices, batch_idx, cfg.batch_size) #feat7,
        # 배치 사용자 단위로 학습
        user_ids = torch.LongTensor(user_ids).to(cfg.device)
        item_ids = torch.LongTensor(item_ids).to(cfg.device)
        feat0 = torch.FloatTensor(feat0).to(cfg.device)
        feat1 = torch.FloatTensor(feat1).to(cfg.device)
        feat2 = torch.LongTensor(feat2).to(cfg.device)
        feat3 = torch.LongTensor(feat3).to(cfg.device)
        feat4 = torch.FloatTensor(feat4).to(cfg.device)
        feat5 = torch.LongTensor(feat5).to(cfg.device)
        feat6 = torch.FloatTensor(feat6).to(cfg.device)
        # feat7 = torch.FloatTensor(feat7).to(cfg.device)
        labels = torch.FloatTensor(labels).to(cfg.device)
        labels = labels.view(-1, 1)

        # grad 초기화
        optimizer.zero_grad()

        # 모델 forward
        output = model.forward(user_ids, item_ids, [feat0, feat1, feat2, feat3, feat4, feat5, feat6])# , feat7
        if output is not None:
          output = output.view(-1, 1)

          loss = criterion(output, labels)

          # 역전파
          loss.backward()

          # 최적화
          optimizer.step()    
          if torch.isnan(loss):
              print('Loss NAN. Train finish.')
              break
          curr_loss_avg = update_avg(curr_loss_avg, loss, step)
          
          msg = f"epoch: {cfg.epoch}, "
          msg += f"loss: {curr_loss_avg.item():.5f}, "
          msg += f"lr: {optimizer.param_groups[0]['lr']:.6f}"
          bar.set_description(msg)
        else:
          pass
    rets = {'losses': np.around(curr_loss_avg.item(), 5)}
    return rets