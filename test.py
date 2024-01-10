
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


def valid_epoch(cfg, u_features, i_features, search_buy_dicts, model, data, mode='valid'):
    pred_list = []
    model.eval()
    search_dict, buy_dict, buy = search_buy_dicts
    user_features, user_features_sex, user_features_views = u_features
    item_features, item_features_run = i_features

    query_user_ids = data['profile_id'].unique() # 추론할 모든 user array 집합
    full_item_ids = np.array([c for c in range(cfg.n_items)]) # 추론할 모든 item array 집합 
    # full_item_ids_feat1 = [item_features['genre_mid'][c] for c in full_item_ids]
    full_item_ids_feat3 = [item_features['genre_mid'][c] for c in full_item_ids]
    full_item_ids_feat4 = [item_features_run['run_time'][c] for c in full_item_ids]
    # cnt = CountVectorizer(min_df=1, ngram_range = (1,1))
    # df_cnt = cnt.fit_transform(meta_new[meta_new['album_id']<cfg.n_items]['features'].values)
    # a_matrix = pd.DataFrame(df_cnt.todense(), columns = cnt.get_feature_names())
    for user_id in query_user_ids:
        # 텐서에 연산 기록을 중지하고, 학습된 모델로 inference하는 과정
        with torch.no_grad():
          user_ids = np.full(cfg.n_items, user_id)
          
          user_ids = torch.LongTensor(user_ids).to(cfg.device)
          item_ids = torch.LongTensor(full_item_ids).to(cfg.device)
          
          feat0 = np.full(cfg.n_items, user_features['age'][user_id])
          feat0 = torch.FloatTensor(feat0).to(cfg.device)
          feat1 = np.full(cfg.n_items, user_features_views['views'][user_id])
          feat1 = torch.FloatTensor(feat1).to(cfg.device)
          feat2 = np.full(cfg.n_items, user_features_sex['sex'][user_id])
          feat2 = torch.LongTensor(feat2).to(cfg.device)
          feat3 = torch.LongTensor(full_item_ids_feat3).to(cfg.device)
          feat4 = torch.FloatTensor(full_item_ids_feat4).to(cfg.device)
          # detach로 validation set에 대한 그래프 history 차단 후 inference
          feat5 = np.zeros(cfg.n_items)
          if user_id in search_dict.keys():
            np.put(feat5, search_dict[user_id], 1)
          feat5 = torch.LongTensor(feat5).to(cfg.device)           
          feat6 = np.zeros(cfg.n_items)
          if user_id in buy_dict.keys():
            np.put(feat6, buy_dict[user_id], buy[(buy['profile_id']==user_id)&(buy['album_id']==buy_dict[user_id][0])]['payment'].values[0])
          feat6 = torch.FloatTensor(feat6).to(cfg.device)
          # feat7_lst = keyword_embedding
          # feat7 = torch.FloatTensor(feat7_lst).to(cfg.device)
          eval_output = model.forward(user_ids, item_ids, [feat0, feat1, feat2, feat3, feat4, feat5, feat6]).detach().cpu().numpy()#, feat7
          pred_u_score = eval_output.reshape(-1)   
        

        pred_u_idx = np.argsort(pred_u_score)[::-1]         # 최종 스코어 높은 순서대로 인덱스 뽑기
          
        pred_u = full_item_ids[pred_u_idx]                  # 전체 아이템 중에서 스코어가 높은 인덱스들 뽑아 리스트 가져오기

        pred_list.append(list(pred_u[:cfg.top_k]))          # top_k개 만큼을 pred_list에 넣기

    pred = pd.DataFrame()
    pred['profile_id'] = query_user_ids
    pred['predicted_list'] = pred_list
    
    # 모델 성능 확인 
    if mode == 'valid':
        rets = evaluation(data, pred)
        return rets, pred
    return pred