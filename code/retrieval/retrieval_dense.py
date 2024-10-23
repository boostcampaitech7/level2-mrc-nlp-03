import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util

"""
SentenceTransformer를 활용한 Bi-Encoder Dense Retrieval
"""

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class DenseRetrieval:
    def __init__(
        self,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        data_path: Optional[str] = "../../data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            model_name:
                SentenceTransformer 모델 이름입니다.
            data_path:
                데이터가 보관되어 있는 경로입니다.
            context_path:
                문서들이 저장된 파일명입니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # 중복 제거를 위한 dict.fromkeys
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # SentenceTransformer 모델 로드
        self.model = SentenceTransformer(model_name)

        self.p_embedding = None  # get_dense_embedding()로 생성합니다

    def get_dense_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            임베딩을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # 임베딩을 저장합니다.
        pickle_name = f"dense_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle loaded.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.model.encode(
                self.contexts, convert_to_tensor=True, show_progress_bar=True
            )
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 10
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str 또는 Dataset 형태의 질의를 받습니다.
            topk (Optional[int], optional): Defaults to 10.
                상위 몇 개의 문서를 반환할지 지정합니다.

        Returns:
            단일 질의의 경우 -> Tuple(List, List)
            다중 질의의 경우 -> pd.DataFrame
        """

        assert self.p_embedding is not None, "먼저 get_dense_embedding()을 실행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print(f"\n[Query]\n{query_or_dataset}\n")
            for i in range(topk):
                print(f"Top-{i+1} passage (Score: {doc_scores[i]:.4f})")
                print(self.contexts[doc_indices[i]])
            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("Bulk query search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example and "answers" in example:
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)
            return pd.DataFrame(total)

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                단일 질의를 받습니다.
            k (Optional[int]): Defaults to 1
                상위 몇 개의 문서를 반환할지 지정합니다.
        """
        query_vec = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_vec, self.p_embedding)[0].cpu().numpy()
        print(scores)
        topk_idx = np.argsort(scores)[::-1][:k]
        return scores[topk_idx], topk_idx

    def get_relevant_doc_bulk(
        self, queries: List[str], k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List[str]):
                다중 질의를 받습니다.
            k (Optional[int]): Defaults to 1
                상위 몇 개의 문서를 반환할지 지정합니다.
        """
        query_vecs = self.model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
        scores = util.dot_score(query_vecs, self.p_embedding).cpu().numpy()
        doc_scores = []
        doc_indices = []
        for i in range(len(queries)):
            topk_idx = np.argsort(scores[i])[::-1][:k]
            doc_scores.append(scores[i][topk_idx])
            doc_indices.append(topk_idx)
        return doc_scores, doc_indices

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../../data/train_dataset", metavar="../../data/train_dataset", type=str, help="Path to the dataset"
    )
    parser.add_argument(
        "--model_name",
        metavar="paraphrase-multilingual-MiniLM-L12-v2",
        type=str,
        default="paraphrase-multilingual-mpnet-base-v2", # paraphrase-multilingual-mpnet-base-v2
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
    print(f"Dataset size: {len(full_ds)}")

    # DenseRetrieval 인스턴스 생성 및 임베딩 생성
    retriever = DenseRetrieval(
        model_name=args.model_name,
        data_path=args.data_path,
        context_path=args.context_path,
    )
    retriever.get_dense_embedding()

    # 단일 질의 테스트
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    # retriever.retrieve(query, topk=3)

    # 다중 질의 테스트
    df = retriever.retrieve(full_ds, topk=40)
    if "original_context" in df.columns:
        df["correct"] = df.apply(lambda row: row["original_context"] in row["context"], axis=1)
        accuracy = df["correct"].mean()
        print(f"Accuracy: {accuracy * 100:.2f}%")
