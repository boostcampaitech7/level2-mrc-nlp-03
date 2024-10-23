import json
import os
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import trange
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
        
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class HybridRetrieval:
    def __init__(
        self,
        model_name_or_path,
        tokenizer_name_or_path,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Summary:
            패시지를 로드하고, 스파스한 검색을 위한 TF-IDF 벡터라이저를 초기화하며, 크로스 인코더 모델과 토크나이저를 로드합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # 중복 제거
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # 스파스한 검색을 위한 TF-IDF 벡터라이저 초기화 및 임베딩 생성
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.p_embedding = self.vectorizer.fit_transform(self.contexts)

        # 크로스 인코더 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=1  # 회귀를 위해 출력 레이블 수를 1로 설정
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 5, sparse_topk: Optional[int] = 50
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Summary:
            각 쿼리에 대해 TF-IDF를 사용하여 상위 N개의 패시지를 검색한 후, 크로스 인코더로 재정렬하여 상위 K개의 패시지를 반환합니다.
        """

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, topk=topk, sparse_topk=sparse_topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # 패시지를 검색하고 DataFrame으로 반환
            total = []
            with timer("Hybrid retrieval"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], topk=topk, sparse_topk=sparse_topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Hybrid retrieval: ")
            ):
                tmp = {
                    # 쿼리와 해당 id 반환
                    "question": example["question"],
                    "id": example["id"],
                    # 검색된 패시지 반환
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # 정답이 있으면 포함
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_sparse_topk(self, query: str, k: int):
        # 쿼리 벡터화
        query_vec = self.vectorizer.transform([query])
        # 유사도 계산
        result = query_vec * self.p_embedding.T
        result = result.toarray().squeeze()
        # 상위 k개 선택
        sorted_result = np.argsort(result)[::-1]
        doc_scores = result[sorted_result][:k]
        doc_indices = sorted_result[:k]
        return doc_scores, doc_indices

    def train_with_inbatch_negatives(
        self,
        dataset,
        batch_size: int = 32,
        num_epochs: int = 1,
        learning_rate: float = 5e-5,
    ):
        """
        Summary:
            KorQuAD 데이터셋의 question과 context를 이용하여 in-batch negatives로 크로스 인코더 모델을 학습합니다.
        """
        ## 누가 학습 좀 시켜줘요
        pass

    def get_relevant_doc(self, query: str, topk: Optional[int] = 5, sparse_topk: Optional[int] = 50) -> Tuple[List, List]:

        """
        Summary:
            쿼리에 대해 TF-IDF로 상위 N개의 패시지를 선택하고, 크로스 인코더를 적용하여 상위 K개의 패시지를 반환합니다.
        """

        # 1단계: 스파스한 방법으로 상위 N개의 패시지 선택
        _, candidate_indices = self.get_sparse_topk(query, k=sparse_topk)
        candidate_passages = [self.contexts[idx] for idx in candidate_indices]

        # 2단계: 크로스 인코더로 재정렬
        scores = []
        indices = []

        batch_size = 512  # GPU 메모리에 따라 조정
        with torch.no_grad():
            for start_idx in range(0, len(candidate_passages), batch_size):
                end_idx = min(start_idx + batch_size, len(candidate_passages))
                passages = candidate_passages[start_idx:end_idx]

                inputs = self.tokenizer(
                    [query] * len(passages),
                    passages,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze()  # (batch_size,)
                scores.extend(logits.cpu().numpy())
                indices.extend(candidate_indices[start_idx:end_idx])

        # 상위 topk개 선택
        scores = np.array(scores)
        indices = np.array(indices)
        sorted_indices = np.argsort(scores)[::-1]
        topk_scores = scores[sorted_indices][:topk]
        topk_indices = indices[sorted_indices][:topk]

        return topk_scores.tolist(), topk_indices.tolist()

    def get_relevant_doc_bulk(
        self, queries: List[str], topk: Optional[int] = 5, sparse_topk: Optional[int] = 50
    ) -> Tuple[List, List]:

        """
        Summary:
            여러 쿼리에 대해 TF-IDF로 상위 N개의 패시지를 선택하고, 크로스 인코더를 적용하여 상위 K개의 패시지를 반환합니다.
        """

        doc_scores = []
        doc_indices = []

        for query in tqdm(queries, desc="Processing queries"):
            topk_scores, topk_indices = self.get_relevant_doc(query, topk=topk, sparse_topk=sparse_topk)
            doc_scores.append(topk_scores)
            doc_indices.append(topk_indices)

        return doc_scores, doc_indices


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../../data/train_dataset", metavar="../../data/train_dataset", type=str, help="Path to the dataset"
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="monologg/koelectra-base-v3-finetuned-korquad",
        type=str,
        default="monologg/koelectra-base-v3-finetuned-korquad",
        help="",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        metavar="monologg/koelectra-base-v3-finetuned-korquad",
        type=str,
        default="monologg/koelectra-base-v3-finetuned-korquad",
        help="",
    )
    parser.add_argument("--data_path", default="../../data", metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path", default="wikipedia_documents.json", 
        metavar="wikipedia_documents.json", type=str, help=""
    )

    args = parser.parse_args()

    # 데이터셋 로드
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    retriever = HybridRetrieval(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    # 모델 학습
    retriever.train_with_inbatch_negatives(
        dataset=full_ds, batch_size=8, num_epochs=3, learning_rate=5e-5
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    with timer("Hybrid retrieval on full dataset"):
        df = retriever.retrieve(full_ds, topk=5, sparse_topk=50)
        if "original_context" in df.columns:
            df["correct"] = df.apply(lambda row: row["original_context"] in row["context"], axis=1)
            print(
                "correct retrieval result by hybrid retrieval",
                df["correct"].sum() / len(df),
            )
        else:
            print(df.head())

    with timer("Single query retrieval"):
        scores, contexts = retriever.retrieve(query, topk=3, sparse_topk=50)
        for i, (score, context) in enumerate(zip(scores, contexts)):
            print(f"\nTop-{i+1} passage with score {score}:")
            print(context)
