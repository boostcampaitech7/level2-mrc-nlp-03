# ret_test
이 디렉토리는 다양한 리트리버 방식을 테스트하고 성능을 확인하기 위해 사용됩니다. 

## 디렉토리 구조
```
ret_test
├── exp_data/
│   ├── unq_wikipedia_documents.json    # 중복 제거 처리한 wiki json 파일 (용량 문제로 첨부하지 않음)         
│   └── unq_train_dataset.py            # 실험에 사용할 train dataset query 파일
├── exp_result/                         # 실험 결과 저장 경로
├── pickle/                             # 토크나이징 결과 저장 경로
└── ret_exp.ipynb                       # 최종 실험 노트북 파일
```

## 사용 방법
1. 데이터 준비
    - `exp_data/` 디렉토리에 `unq_wikipedia_documents.json` 파일을 추가합니다
2. 필요 함수 호출
    - `ret_exp.ipynb` 파일 (1), (2)의 함수를 호출합니다. 
    - 토크나이징 결과 저장 옵션을 사용할 경우 (3)의 함수도 호출합니다.
3. 사용 토크나이저 호출
    - 사용할 토크나이저를 (4)에서 호출합니다.
    - 본 코드에 존재하지 않는 경우, 직접 추가하여 사용 가능합니다.
4. 실험 실행
    - `perform_experiment` 에 실험할 파라미터를 적용하여 실험을 진행합니다. 
    - 토크나이징 결과를 저장하거나 이미 저장된 결과를 활용한다면 `perform_experiment_with_documents_pickle` 를 사용합니다.
    - 자세한 파라미터에 대한 설명은 함수에서 확인할 수 있습니다. 
5. 결과 분석
    - topk에 따른 리트리버 성능을 jupyter notebook 내에서 확인할 수 있습니다. 
    - `exp_result/` 디렉토리에 저장된 결과 파일로 세부 결과를 확인합니다.
    - failed query가 실제로 예측한 문서 번호도 확인할 수 있습니다.
