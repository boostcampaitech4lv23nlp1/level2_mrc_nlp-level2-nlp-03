from typing import List, Optional, Tuple
import numpy as np
from tqdm.auto import tqdm # 사실 tqdm을 사용하지 않아도 됩니다

def get_relevant_doc_es(
    queries: str,
    es,
    k: Optional[int] = 1, 
    bulk:bool=False) -> Tuple[List, List]:
    """
    Arguments:
        query (str): 하나의 Query를 받습니다.
        k (Optional[int] = 1): 상위 몇 개(Top-k)의 Passage를 반환할지 정합니다.
        bulk (bool = False): 여러개의 문서를 반환할지 제어합니다.
    Note:
        vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
    """
    if bulk:
        doc_score = []
        doc_indices = []

        for query in tqdm(queries):
            res = es.search(index = "document",q=query, size=k)
            doc_score.append([hit['_score'] for hit in res['hits']['hits']])
            doc_indices.append([int(hit['_id']) for hit in res['hits']['hits']])
    
    else:
        res = es.serach(index='document',q=queries,size=k)
        doc_score = res['hits']['hits']['_score']
        doc_indices = res['hits']['hits']['_id']

    return doc_score, doc_indices

def get_relevant_doc_faiss(
    query: str,
    tfidfv,
    indexer,
    k: Optional[int] = 1,
    bulk: bool=False) -> Tuple[List, List]:
    """
    Arguments:
        query (str):
            하나의 Query를 받습니다.
        k (Optional[int] = 1):
            상위 몇 개의 Passage를 반환할지 정합니다.
        bulk (bool = False): 여러개의 문서를 반환할지 제어합니다.
    Note:
        vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
    """

    if bulk and isinstance(query, list): # bulk(여러 passage)
        query_vec = tfidfv.transform(query)

    elif not bulk and isinstance(query, str): # not bulk(단일 passage)
        query_vec = tfidfv.transform([query])

    assert (np.sum(query_vec) != 0), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

    q_emb = query_vec.toarray().astype(np.float32)
    D, I = indexer.search(q_emb, k)

    if bulk:
        return D.tolist(), I.tolist()

    elif not bulk:
        return D.tolist()[0], I.tolist()[0]