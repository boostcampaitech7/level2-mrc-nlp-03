## 🍗반반치킨 Team

|[정준한](https://github.com/??)|[이수진](https://github.com/??)|[육지훈](https://github.com/jihunyuk)|[전진](https://github.com/jeenie2727)|[이금상](https://github.com/??)|[허윤서]([https://github.com/??](https://github.com/Yunseo-Lab))|
|:-:|:-:|:-:|:-:|:-:|:-:|
|<a href="https://github.com/??"><img src="profile/바로핑.png" width='300px'></a>|<a href="https://github.com/??"><img src="profile/차차핑.png" width='300px'></a>|<a href="https://github.com/jihunyuk"><img src="profile/하츄핑.png" width='300px'></a>|<a href="https://github.com/??"><img src="profile/라라핑.png" width='300px'></a>|<a href="https://github.com/??"><img src="profile/해핑.png" width='300px'></a>|<a href="[https://github.com/??](https://github.com/Yunseo-Lab)"><img src="profile/아자핑.png" width='300px'></a>|

## Members' Role
| Member | Role | 
| --- | --- |
| 정준한 | EDA, 코드 모듈화, BM25 추가, 도메인 적응 코드 추가 |
| 이수진 | BM25 Plus추가, Pre-trained 모델 실험, 리트리버 성능 실험 |
| 육지훈 | EDA, 하이퍼파라미터 서치, 성능 검증 코드 제작, Inference 후처리 |
| 전진 | EDA, 리트리버 실험 설계, 리트리버 성능 개선  |
| 이금상 | EDA, Pre-trained 모델 실험, 리트리버 성능 실험 |
| 허윤서 | Retrieved data EDA, Cross, Bi-encoder DPR & Re-rank |

### 0. 설치

- pre-commit을 위해 아래 명령어를 실행해주세요.
- code/requirements.txt를 잘 확인해주세요.
```Bash
$ pip install pre-commit
$ pip install omegaconf # yaml 파일로 인한 모듈화로 추가 (requirements.txt에도 추가해둠)
$ pip install rank-bm25 # bm25 설치 (requirements.txt에도 추가해둠)
```

### 1. 최근 Branch 변경사항
- base_config.yaml을 이용해 조정 가능
- Retriever: BM25OKapi, BM25plus 추가
- EDA 코드 추가
- config에서 newline_to_space 인자로 train시 전처리 적용유무 선택 가능
- config에서 data_type 인자로 KorQuAD 1.0 사용할지, 기존 데이터셋 이용할지 선택 가능
- train_dataset_name에 csv파일도 넣으면 불러올수 있도록 기능 구현
- config에 retrieval_tokenizer로 retrieve할 때 사용할 토크나이저 선택 가능
- 로그는 스텝 단위, 저장은 에폭 단위
- fp16으로 빠르게 학습, 추론
- wandb에 entity인자로 팀 프로젝트 경로 설정
- huggingface에 모델 올려서 공유


### 2. 사용법
1. config/base_config.yaml 파일을 수정합니다.
2. 사용할 config파일을 뒤에 붙여서 파일을 실행시킵니다.
```Bash
$ python train.py --config_path config/base_config.yaml
$ python inference.py --config_path config/base_config.yaml
```

### 3. 참고사항
- roberta-large로 훈련시킬 경우, 2에폭 기준(batch 16) 14분 정도 걸렸던 것 같습니다.
- BM25로 scores를 반환하는데 걸리는 시간은 약 4분입니다.  
  retrieval할 때마다, 4분이 걸려서 scores 자체를 저장하는 방식도 고려해봐야 될 것 같습니다.  
  현재, BM25OKapi 객체를 pickle로는 저장할 수 있도록 코드를 구현한 상태입니다.
- roberta-large로 추론할 경우, topk=40을 모두 이어붙였더니 추론시간이 약 19분 걸립니다.
- 모듈화는 5기생 5조 github 코드에서 아이디어를 가져왔고 코드를 참고하였습니다. BM25도 5조 github 코드를 참고하여 필요한 부분만 참고하였습니다. 다만, 모든 코드는 베이스라인 위에서 직접 수정하며 구현했습니다. prepare_dataset.py도 5조 아이디어에서 착안.
- roberta-large에 KorQuAD 1.0 1에폭 훈련시켰더니 훈련시간 1시간 나옵니다. (배치 16 기준)
- 1차로는 korquad, max_epoch 1, learning_rate 3e-5, step설정 1000으로
- 2차로는 original, max_epoch 4, learning_rate 9e-6, step설정 500으로 해서 진행하는게 좋을 것 같은 느낌..

### 4. config 설정법
```YAML
model:
  model_name_or_path: klue/roberta-large # huggingface에서 불러올 모델 or 저장된 모델 경로 # models/roberta_original
  config_name: null # 변경 x
  tokenizer_name: null # 변경 x
  retrieval_tokenizer: monologg/koelectra-base-v3-finetuned-korquad # BM25로 retrieve할때만 쓰는 토크나이저 이름

data:
  train_dataset_name: ../data/train_dataset # 훈련시 사용할 데이터셋 경로. csv파일 경로도 가능하다. # ../data/train_dataset.csv
  inference_dataset_name: ../data/test_dataset # 변경 x
  context_path: ../data/korean_ratio_0.40_up.json # context로 사용할 위키데이터셋 경로 # wikipedia_documents.json
  overwrite_cache: False # 변경 x
  preprocessing_num_workers: null # 변경 x
  max_seq_length: 384 # 모델이 받아들이는 최대 context. # 512까지 가능하나 시간이 길어짐.
  pad_to_max_length: False # 변경 x
  doc_stride: 128 
  max_answer_length: 100
  eval_retrieval: True # 변경 x
  num_clusters: 64 # 변경 x
  top_k_retrieval: 40 # topk
  use_faiss: False # 변경 x
  retrieval_type: bm25Plus # [tfidf, bm25, bm25Plus]에서 택1. 사용할 retriever 종류.
  data_type: original # [original, korquad, korquad_hard] 중 택1. 자세한 것은 prepare_dataset.py 참조. train시 Korquad 1.0과 기존 데이터셋 중 사용할 데이터셋 선택 가능.
  newline_preprocess: space # [remove, space] 중 택1. remove일 경우, train할 때 context를 \\n을 공백으로 바꿔주고 train함 (data_type: original을 설정할때만 적용됨. korquad는 True로 해도 데이터셋 전처리를 진행하지 않음)
  
train:
  batch_size: 16
  max_epoch: 4
  learning_rate: 3e-5 #3e-5 #9.0e-6
  eval_step: 500 # 훈련 진행시 evaluate을 하는 간격
  logging_step: 500 # 1000 # 500 # 훈련 진행시 wandb에 로그가 찍히는 간격
  save_step: 500 # 훈련 진행시 모델을 저장하는 간격인데 사용안함
  gradient_accumulation: 1
  do_train: True
  do_eval: True
  do_predict: False
  train_output_dir: models/roberta-large # 훈련한 모델이 저장될 경로
  inference_output_dir: outputs/roberta-large # 추론 진행시 predictions.json이 저장되는 경로.
  seed: 42
  save_total_limit: 1 # 학습 중 저장할 모델의 최대 개수. 설정된 개수만큼 저장되면, 새로운 모델을 저장할 때 성능이 좋은 모델을 남기고 성능이 떨어지는 모델은 자동으로 삭제됨.
  overwrite_output_dir: False # train_output_dir이 이미 존재할 때, 덮어쓸지 여부. False로 설정하고, 이미 폴더가 있을 경우 에러발생.
  fp16: True # fp16을 사용할지 여부. True로 하면 속도가 더 빨라지고 가벼워짐.(성능하락은 별로 없어서 True로 고정)

wandb:
  use: True # wandb 사용여부. 추론할 때는, True로 해도 wandb 사용 안함.
  entity: halfchicken_p2 # 팀 이름
  project: odqa_finetuning
  name: name # 개인 이름으로 설정. 이름 뒤에 model_name_epoch_bs_learning_rate이 붙음.
```

### 5. 디렉토리 구조
```Bash
level2-mrc-nlp-03/code/
|
|-- assets
|
|-- config # config 모음
|   |-- base_config.yaml
|
|-- models # 모델 저장하는 경로(train_output_dir)
|
|-- EDA
|
|-- outputs/ # 추론 결과물 경로(inference_output_dir)
|
|-- retrieval/ # Retrieve할때 클래스 모음
|   |-- retrieval_bm25.py
|   |-- retrieval_bm25Plus.py
|   |-- retrieval_sparse.py
|
|-- ret_test/ # Retrieve 실험용 코드 모음

|-- wandb
|
|-- inference.py
|-- prepare_dataset.py # config에서 설정한 data_type에 따라 dataset을 반환해줌 (KorQuAD 1.0 사용할지, 기존 데이터셋 이용할지 등..)
|-- requirements.txt
|-- train.py
|-- trainer_qa.py
|-- utils_qa.py
|-- utils.py # \\n을 공백으로 전환하는 함수, csv에서 데이터셋 불러올때 str->파이썬 객체로 변환해주는 함수
```
```Bash
level2-mrc-nlp-03/data/
|
|-- test_dataset
|-- train_dataset
|-- bm25_model.bin # 추론시 저장되는 bm25Okapi 객체
|-- bm25_plus_model.bin # 추론시 저장되는 bm25plus객체
|-- korean_ratio_0.40_up.json # 전처리한 위키피디아 json파일
|-- wikipedia_documents.json
|-- bm25_plus_scores_indices.bin # 추론시 저장되는 indices, scores 피클파일
|-- bm25_scores_indices.bin # 추론시 저장되는 indices, scores 피클파일
```

> 에러 발생시 Issue로 빠르게 알려주시면 감사하겠습니다.   
> 실험결과나 아이디어 관련해서는 Notion을 참고해주세요.  
> 해당 실험시, config와 결과값, 실행방법은 모두 Notion에 적어두었습니다.
