### 0. 설치

- pre-commit을 위해 아래 명령어를 실행해주세요.
- code/requirements.txt를 잘 확인해주세요.
```Bash
$ pip install pre-commit
$ pip install omegaconf # yaml 파일로 인한 모듈화로 추가 (requirements.txt에도 추가해둠)
$ pip install rank-bm25 # bm25 설치 (requirements.txt에도 추가해둠)
```

### 1. 최근 Branch 변경사항
- base_config.yaml 방식으로 변경 (기존 arguments.py 삭제)
- 실행시킬 때, --config_path를 뒤에 붙여서 적용할 config 선택 가능
- BM25Retrieval 방식 추가


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
- 모듈화는 5기생 5조 github 코드에서 아이디어를 가져왔고 코드를 참고하였습니다. BM25도 5조 github 코드를 참고하여 필요한 부분만 참고하였습니다. 다만, 모든 코드는 베이스라인 위에서 직접 수정하며 구현했습니다.

### 4. config 설정법
- model_name_or_path: 사용할 모델명이나 사용할 모델 경로
- retrieval_tokenizer: BM25로 retrieve할때만 쓸 토크나이저명
- 그 외 설정들은 알아서 잘 조절하세요.

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
|-- outputs/ # 추론 결과물 경로(inference_output_dir)
|
|-- retrieval/ # Retrieve할때 클래스 모음
|   |-- retrieval_bm25.py
|   |-- retrieval_sparse.py
|
|-- wandb
|
|-- inference.py
|-- requirements.txt
|-- train.py
|-- trainer_qa.py
|-- utils_qa.py
```


> 에러 발생시 Issue로 빠르게 알려주시면 감사하겠습니다.