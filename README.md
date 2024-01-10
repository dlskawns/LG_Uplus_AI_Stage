# LG Uplus 아이들나라 OTT 추천시스템 경진대회

최종 모델 아키텍쳐

<img src = 'https://i.imgur.com/vm3tipQ.png'>


## 프로젝트 배경 및 목표
**OTT서비스의 콘텐츠 추천 시스템 알고리즘 개발**
* 목표: LG U플러스 ‘아이들나라’의 고객 행동데이터 기반 추천 모델의 성능을 향상하는 것을 목표로 하는 대회입니다. 콘텐츠 추천을 위한 추천시스템 알고리즘 개발 및 성능 개선을 목표합니다.


## 진행 구성 및 결과
* 팀 구성: 2인 1팀 (사정상 혼자 진행)
* 모델 처리 방안: NeuMF에 주요 feature 추가
* 최종 결과: 종합스코어(Recall/NDCG) 0.2204
* 순위: 35등 - 총 658팀 중 상위 5% 이내


## 프로젝트 내용
### 1. EDA
- 전체 데이터 테이블 컬럼 요소 확인
    1. history_data.csv - 시청시작 데이터(상호작용)(1005651, 8)
    2. profile_data.csv - 유저 메타 데이터 (8311, 9)
    3. meta_data.csv - 아이템 메타 데이터 (42602, 16)
- 데이터 분포, 결측치 등 기본 데이터 정보 확인 (데이터 공개 불가)

### 실험 가설 정립 (Model Feature Selection process)
* 연령 (User meta data) / 장르 (Genre) -> 데이터 공개 불가로 시각자료 상 수치 제거
  <img src='https://i.imgur.com/8n46KFm.png'>

    - 연령과 활성사용 분포가 다른 만큼 유저 별 세분화된 추천이 필요
        - ‘age’ feature를 추가적으로 고려해볼 것
    - 연령(age)에 따른 장르(genre) 선호여부 확인을 위한 $\chi^2$ 동질성 검정 시행
        - $h_0:$ 집단(연령) 간 선호하는(view 수 기준)장르 분포가 같다.
        - $h_1:$ 집단(연령) 간 선호하는 장르 분포가 다르다.
        - 결과:
            - $\chi^2$ 검정 통계량: 298216.81
            - P-value = 0.0
            귀무가설을 기각하고, 분포가 다름을 확인 → **연령 별 장르에 대한 고려가 필요.**
* 런타임 (Item meta data) / 아이템 소비 횟수(인기의 정도)
  <img src= 'https://i.imgur.com/qqdD85F.png'>

    - 산점도를 통한 선형성 확인 → 특정할 수 없으며, 런타임 및 총사용 수 변수의 왜도가 높아 log변환을 통한 상관분석 동시 시행
    - 변환 이전:
        - 피어슨 상관계수: -0.168 / P-value = 0.0
        - 스피어만 순위상관계수: -0.205 / P-value = 0.0
    - 변환 이후:
        - 피어슨 상관계수: -0.206 / P-value = 0.0
        - 스피어만 순위상관계수: -0.205 / P-value = 0.0
    - 결과:
        - 음의 상관성이 미세하게 존재하므로, 변수 추가하여 실험진행
### 3. 모델 실험
- 베이스라인 모델
    - Model: NeuMF
    - Model Architecture
        <img src='https://i.imgur.com/DB89XUq.png'>
        - 좌측(in blue) - 유저, 아이템 간 상호작용 여부 파악을 하는 Matrix Factorization 모델
        - 우측(in purple) - 유저, 아이템 상호작용 이외의 특성을 반영하여 비선형 적으로 스코어를 계산하는 DNN 모델
        - 최상위 Layer에서 각 score 결과값을 합산해 최종 Score를 계산하는 방식
- 실험 모델 1: 특성 추가 반영
    - Model Architecture
        <img src='https://i.imgur.com/Bh6ptPY.png'>
        - 특성 실험 1: 런타임, 시청 수(Views) 데이터 추가
            - 런타임(run_time)이 수요에 영향을 미치는 점을 고려하여 추가 실험진행
            - 시청 수(views)정보를 통해 유저 정보를 추가적으로 고려하여 실험 진행
            - 성별(sex)정보를 통해 유저 정보를 추가적으로 고려하여 실험 진행
        - 특성 실험 2: 런타임 및 상호작용 특성 (결제여부, 검색여부)
            - 모델 실험 가설: 결제|검색을 했던 경우 재시청을 하는 경우 존재하므로 관련 Feature 추가
- 실험 모델 2: 계산 레이어 추가 합산 스코어 계산
    - Model Architecture
        <img src='https://i.imgur.com/CupWvCN.png'>
            - Deep FM(Deep + Factorization Machine)에서 차용
            - FM Layer를 통해 추가적으로 Features 간의 관계를 파악하도록 처리
### 4. 성능 평가
- ValidationSet Score 기준 평가 metric
    - Baseline
      
        <img src='https://i.imgur.com/cDlOPH9.png'>
        
    - Opt1: Baseline + runtime + sex + view
      
        <img src = 'https://i.imgur.com/Kj4SqoI.png'>
        
    - Opt2: Baseline + runtime + sex + view + search + payment
      
        <img src = 'https://i.imgur.com/Qfr1ssx.png'>
        
    - Opt3: Baseline + Full Features + FM Layer (GPU 문제로 10회)
      
        <img src = 'https://i.imgur.com/C3iI1b7.png'>
        
    - Public Score 기준 평가 metrics (더 중요)
        - Total Score(Rank)
        - Recall@25
        - NDCG@25
    - 결과
        |Model|Score|Recall@25|NDCG@25|
        |:-:|:-:|:-:|:-:|
        |baseline|0.2136|0.2181|0.2001|
        |Opt1|0.2150|0.2198|0.2004|
        |Opt2|0.2204|0.2251|0.2065|
        |Opt3|0.2109|0.2157|0.1967|

        * FM 레이어를 함께 사용한 모델은 성능이 많이 떨어져서 Opt2모델을 최종 모델로 선정하였음

## 회고 및 느낀 점
    **사실 및 개선점** 
    
    1. 특성공학과 연관성 분석에 초점을 너무 두어 다른 모델을 구현하거나 내실을 다지는 데에는 실패했습니다.
        a. 추후에는 초기 설정한 실험 계획과 타임라인에 맞춰 진행하고, 이후에 개선할 것을 목표로 할 것입니다.
    2. 팀을 구성하지 않아 너무 늦게 진행하게 되었는데, 팀을 구성해서 진행한다면 좀 더 진도를 빠르게 가져가면서 다양한 실험을 해볼 수 있을 것으로 예상됩니다.

    **향후 모델 개선 방안**

    1. CV를 통한 모델 정확도 향상 시키는 방향
    2. 레이어의 조정 레이어 추가 및 Dropout, ReLU 등의 하이퍼 파라미터 조정
    3. 유저 세그먼테이션을 통한 계층 별 모델 구성
