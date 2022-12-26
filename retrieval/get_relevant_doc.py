from typing import List, Optional, Tuple
import numpy as np

def get_relevant_doc(
    query: str,
    tfidfv,
    p_embedding,
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
    if bulk and isinstance(query, list): # bulk(여러 passage)
        query_vec = tfidfv.transform(query)

    elif not bulk and isinstance(query, str): # not bulk(단일 passage)
        query_vec = tfidfv.transform([query])

    assert (np.sum(query_vec) != 0), "query에 vectorizer의 vocab에 없는 단어만 존재합니다."

    result = query_vec * p_embedding.T
    if not isinstance(result, np.ndarray): # 만약 넘파이가 아니라면 넘파이 배열로 바꿔주기
        result = result.toarray()

    if bulk:
        doc_score = []
        doc_indices = []
        for i in range(result.shape[0]): # 문서 갯수만큼 반환
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_score.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

    elif not bulk:
        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]

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