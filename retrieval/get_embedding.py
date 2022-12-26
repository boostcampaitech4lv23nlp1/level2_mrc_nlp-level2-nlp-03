import os
import pickle
import numpy as np
import faiss

def get_sparse_embedding(data_path, tfidfv, contexts):
    """
    Summary:
        Passage Embedding을 만들고
        TFIDF와 Embedding을 pickle로 저장합니다.
        만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
    """

    # Pickle을 저장합니다.
    pickle_name = f"sparse_embedding.bin"
    tfidfv_name = f"tfidv.bin"
    emd_path = os.path.join(data_path, pickle_name)
    tfidfv_path = os.path.join(data_path, tfidfv_name)

    if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
        with open(emd_path, "rb") as file:
            passage_embedding = pickle.load(file)
        with open(tfidfv_path, "rb") as file:
            tfidfv = pickle.load(file)
        print("Embedding pickle load.")
    else:
        print("Build passage embedding")
        passage_embedding = tfidfv.fit_transform(contexts)
        print('passage_embedding.shape', passage_embedding.shape)
        with open(emd_path, "wb") as file:
            pickle.dump(passage_embedding, file)
        with open(tfidfv_path, "wb") as file:
            pickle.dump(tfidfv, file)
        print("Embedding pickle saved.")
    
    return passage_embedding, tfidfv

def build_faiss(data_path, passage_embedding, num_clusters=64):

    """
    Summary:
        속성으로 저장되어 있는 Passage Embedding을
        Faiss indexer에 fitting 시켜놓습니다.
        이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

    Note:
        Faiss는 Build하는데 시간이 오래 걸리기 때문에,
        매번 새롭게 build하는 것은 비효율적입니다.
        그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
        다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
        제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
    """

    indexer_name = f"faiss_clusters{num_clusters}.index"
    indexer_path = os.path.join(data_path, indexer_name)
    if os.path.isfile(indexer_path):
        print("Load Saved Faiss Indexer.")
        indexer = faiss.read_index(indexer_path)

    else:
        p_emb = passage_embedding.astype(np.float32).toarray()
        emb_dim = p_emb.shape[-1]

        num_clusters = num_clusters
        quantizer = faiss.IndexFlatL2(emb_dim)

        indexer = faiss.IndexIVFScalarQuantizer(
            quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
        )
        indexer.train(p_emb)
        indexer.add(p_emb)
        faiss.write_index(indexer, indexer_path)
        print("Faiss Indexer Saved.")