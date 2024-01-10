import pandas as pd
import torch
import os, random
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
# from tqdm.notebook import tqdm
from train import cfg
data_path = './data'

class Data_Loader:
    """
    데이터 불러오는 인스턴스 생성 
    필요한 데이터만 실험에 따라 추가, 수정가능

    DataLoader 객체를 통해 초기 데이터를 불러온다. 
    history_df: OTT 콘텐츠 시청 시작 데이터 
    profile_df: 유저의 프로필 정보
    buy_df: 콘텐츠의 구매 이력 데이터
    search_df: 콘텐츠의 검색을 통한 시청 데이터 
    """
    print("데이터 불러오기 진행")
    history_df = pd.read_csv(os.path.join(data_path, 'history_data.csv'), encoding='utf-8')
    profile_df = pd.read_csv(os.path.join(data_path, 'profile_data.csv'), encoding='utf-8')
    meta_df = pd.read_csv(os.path.join(data_path, 'meta_data.csv'), encoding='utf-8', low_memory=False)
    buy_df = pd.read_csv(os.path.join(data_path, 'buy_data.csv'), encoding='utf-8')
    watch_df = pd.read_csv(os.path.join(data_path, 'watch_e_data.csv'), encoding='utf-8')
    search_df = pd.read_csv(os.path.join(data_path, 'search_data.csv'), encoding='utf-8')
    metap_df = pd.read_csv(os.path.join(data_path, 'meta_data_plus.csv'), encoding='utf-8')

# class cfg: 
#     gpu_idx = 0
#     device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")
#     top_k = 25
#     seed = 42
#     neg_ratio = 100
#     test_size = 0.1
#     batch_size = 256
#     emb_dim = 256
#     layer_dim = 256
#     dropout = 0.05
#     epochs = 15
#     learning_rate = 0.0025
#     reg_lambda = 0
#     check_epoch = 1

class PreProcess:
    def __init__(self):
        """
        전처리 진행 내용
        """
        dl = Data_Loader()
        
        print("전처리 진행")
        h_df = dl.history_df[['profile_id','log_time','album_id']] # history - 사용자 식별값, 시청시각, 소비한 콘텐츠 정보만 남긴다.
        p_df = dl.profile_df[['profile_id','sex','age']]           # profile - 사용자 식별값, 성별, 나이 정보만 남긴다.
        df = pd.merge(h_df, p_df, 'left', on = 'profile_id')         # profile 기준으로 history와 profile을 병합
        df['sex'] = df['sex'].map(lambda x: 1 if x == 'M' else 0)
        
        # meta 정보 추가하기 
        df = pd.merge(df, dl.meta_df, 'left', on = 'album_id')


        # view 수 정보 추가하기 - 해당 콘텐츠를 몇번 소비했는지의 정보
        view_cnts = df.groupby('profile_id')['album_id'].count()     # view_cnts - 사용자 식별값, 소비 콘텐츠 정보로 소비 횟수를 측정
        df['views'] = df['profile_id'].map(lambda x: view_cnts[x])   # 위에서 측정한 횟수 정보를 파생 변수 views로 추가

        # search 정보 추가하기 - 해당 콘텐츠를 검색해서 소비했는지의 정보
        search = dl.search_df[['profile_id','album_id']].drop_duplicates() # search - 사용자 식별값, 콘텐츠 정보의 중복을 제거하여, 검색 이력의 유무 정보 확인
        search['search'] = 1
        df = pd.merge(df, search, 'left', on=['profile_id','album_id']) # 위에서 생성한 검색 여부 정보를 파생 변수 search로 추가


        # payment 구매 이력 정보 추가하기 - 해당 콘텐츠를 구매했는지 파악
        df = pd.merge(df,dl.buy_df[['profile_id','album_id','payment']].drop_duplicates(), 'left', on = ['profile_id','album_id']) 
        df['payment'] = df['payment'].fillna(0)
        df['payment'] = df['payment'].map(lambda x: 0 if x == 0 else 1)

        # 한번 이상 소비한 콘텐츠인 경우 rating에 1 부터 -> negative sampling을 위한 작업
        data = df[['profile_id','log_time','album_id','views']].drop_duplicates(subset=['profile_id', 'album_id', 'log_time']).sort_values(by = ['profile_id', 'log_time']).reset_index(drop = True)
        data['rating'] = 1
        
        # 유저 수와 아이템 수 확인
        cfg.n_users = data.profile_id.max()+1#-len(one_views_user)
        cfg.n_items = data.album_id.max()+1
        cfg.n_genres = dl.meta_df['genre_mid'].nunique()
        
        # views가 1인 데이터는 제외 -> 횟수가 너무 적어 computing만 많고 비효율적 모델을 만들게 됨
        data_new = data.drop(data[data['views']==1].index).reset_index()

        # 학습 및 검증 데이터 분리
        train, valid = train_test_split(
        data_new, test_size=cfg.test_size, random_state=cfg.seed
        )   

        # Matrix로 변환하여 array방식으로 negative sampling용 데이터 생성
        train = train.to_numpy()
        matrix = sparse.lil_matrix((cfg.n_users, cfg.n_items))
        for (idx, p, _, i, r, view) in train:
            matrix[p, i] = r
            
        train = sparse.csr_matrix(matrix)
        self.train = train.toarray()
        self.valid = valid
    # Feature 추출 딕셔너리 생성
    # 유저 특징 정보 추출

        self.df = df.set_index('profile_id')
        self.user_features = self.df[['age']].to_dict()
        self.user_features_sex = self.df[['sex']].to_dict()
        self.user_features_views = self.df[['views']].to_dict()

        # 아이템 특징 정보 추출 
        le = LabelEncoder()
        dl.meta_df['genre_mid'] = le.fit_transform(dl.meta_df['genre_mid'])
        self.item_features = dl.meta_df[['genre_mid']].to_dict()
        self.item_features_run = dl.meta_df[['run_time']].to_dict()

        # 검색 상호작용 딕셔너리 생성
        self.search = search[search['album_id']<cfg.n_items]
        self.search_dict = {}
        for i in self.search['profile_id'].unique():
            self.search_dict[i] = list(self.search[self.search['profile_id']== i]['album_id'].values)

        # 구매 상호작용 딕셔너리 생성
        self.buy = dl.buy_df[['profile_id','album_id','payment']]
        self.buy_dict = {}
        for i in self.buy['profile_id'].unique():
            self.buy_dict[i] = list(self.buy[self.buy['profile_id']== i]['album_id'].values)

    def user_sex_views_feat(self):
        "유저 딕셔너리 추출"
        return self.user_features, self.user_features_sex, self.user_features_views
    
    def item_run_feat(self):
        "아이템 딕셔러니 추출"
        return self.item_features, self.item_features_run
    
    def search_buy_dicts(self):
        "검색, 구매금액 딕셔너리 추출"
        return self.search_dict, self.buy_dict, self.buy

    def make_UIdataset(self, neg_ratio):
        """ 유저별 학습에 필요한 딕셔너리 데이터 생성 
        Args:
            train : 유저-아이템의 상호작용을 담은 행렬 
                ex) 
                    array([[0., 0., 0., ..., 0., 0., 0.],
                            [0., 0., 0., ..., 0., 0., 0.],
                            [0., 0., 0., ..., 0., 0., 0.],
                            ...,
                            [0., 0., 0., ..., 0., 0., 0.],
                            [0., 0., 0., ..., 0., 0., 0.],
                            [0., 0., 0., ..., 0., 0., 0.]])
            neg_ratio : negative sampling 활용할 비율 
                ex) 3 (positive label 1개당 negative label 3개)
        Returns: 
            UIdataset : 유저별 학습에 필요한 정보를 담은 딕셔너리 
                ex) {'사용자 ID': [[positive 샘플, negative 샘플], ... , [1, 1, 1, ..., 0, 0]]}
                    >>> UIdataset[3]
                        [array([   16,    17,    18, ...,  9586, 18991,  9442]),
                        array([5, 5, 5, ..., 5, 5, 5]),
                        array([4, 4, 4, ..., 5, 1, 1]),
                        array([1., 1., 1., ..., 0., 0., 0.])]
        """
        UIdataset = {}
        for user_id, items_by_user in enumerate(self.train):
            UIdataset[user_id] = []
            # positive 샘플 계산 
            # pos item은 0.5 이상인 것들의 index를 찾아 출력
            pos_item_ids = np.where(items_by_user > 0.5)[0]
            # pos 샘플 개수
            num_pos_samples = len(pos_item_ids)

            # negative 샘플 계산 (random negative sampling)
            # neg 샘플 개수 
            num_neg_samples = neg_ratio * num_pos_samples
            neg_items = np.where(items_by_user < 0.5)[0]
            # neg 샘플 중 랜덤 초이스
            neg_item_ids = np.random.choice(neg_items, min(num_neg_samples, len(neg_items)), replace=False)
            # pos, neg 샘플 짝지어서 UIdataset에 넣기
            UIdataset[user_id].append(np.concatenate([pos_item_ids, neg_item_ids]))
            
            # feature 추출 
            features_age = []
            features_views = []
            features_sex = []
            # pos, neg 쌍을 하나씩 뽑고,
            for item_id in np.concatenate([pos_item_ids, neg_item_ids]): 
                # user_features의 user에 따른 나이를 features에 넣기
                features_age.append(self.user_features['age'][user_id])  # float
                features_views.append(self.user_features_views['views'][user_id])  # float
                features_sex.append(self.user_features_sex['sex'][user_id])  # long
            #나이 정보 순서를 append해서 모든 쌍에 추가해주기
            UIdataset[user_id].append(np.array(features_age))
            UIdataset[user_id].append(np.array(features_views))
            UIdataset[user_id].append(np.array(features_sex))

            features_genre = []
            features_run = []
            features_search = []
            features_payment = []
            # features_keywords = []
            # pos와 neg 일렬 리스트로 변환
            for item_id in np.concatenate([pos_item_ids, neg_item_ids]): 
                # item_features의 item에 따른 장르를 features에 넣기
                
                features_genre.append(self.item_features['genre_mid'][item_id])  # long
                features_run.append(self.item_features_run['run_time'][item_id]) # float
                # features_keywords.append(keyword_embedding[item_id])
                if user_id in self.search_dict.keys():
                    if item_id in self.search_dict[user_id]:
                        features_search.append(1)
                    else:
                        features_search.append(0)
                else:
                    features_search.append(0)
                if user_id in self.buy_dict.keys():
                    if item_id in self.buy_dict[user_id]:
                        features_payment.append(int(self.buy[(self.buy['profile_id']==user_id)&(self.buy['album_id']==item_id)]['payment'].values[0]))
                    else:
                        features_payment.append(0)
                else: 
                    features_payment.append(0)
                
            #장르 정보를 append해서 모든 쌍에 추가해주기 -> 상호작용에 feature가 추가된것임
            UIdataset[user_id].append(np.array(features_genre))
            UIdataset[user_id].append(np.array(features_run))
            UIdataset[user_id].append(np.array(features_search))
            UIdataset[user_id].append(np.array(features_payment))
            # if len(features_keywords) == 0:
            #   UIdataset[user_id].append(np.array(features_keywords))
            # else:
            #   UIdataset[user_id].append(features_keywords)
            # label 저장  
            pos_labels = np.ones(len(pos_item_ids))
            neg_labels = np.zeros(len(neg_item_ids))
            UIdataset[user_id].append(np.concatenate([pos_labels, neg_labels]))

        return UIdataset


preprocess = PreProcess()

