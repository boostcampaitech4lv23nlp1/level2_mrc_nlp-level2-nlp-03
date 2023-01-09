from typing import List, Optional, Tuple, Union
import os,re
import json
import pandas as pd
from datasets import Dataset, Features, Value, DatasetDict
from tqdm.auto import tqdm
# from sklearn.feature_extraction.text import TfidfVectorizer
from elasticsearch import Elasticsearch

# elasticsearch는 context embedding을 따로 구할 필요가 없습니다.
# from .get_embedding_bm25 import get_sparse_embedding_bm25, build_faiss 
from .get_relevant_doc_es import get_relevant_doc_es, get_relevant_doc_faiss

class SparseRetrieval_es:
    def __init__(self,
                tokenizer,
                data_path: Optional[str] = "data/",
                context_path: Optional[str] = "wikipedia_documents.json",
                is_faiss: bool = False):

        """
        Arguments:
            tokenizer:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        self.is_faiss = is_faiss

        with open('./data/wikipedia_documents.json', 'r', encoding="utf-8") as f:
            wiki_json = json.load(f)
            wiki_df = pd.DataFrame(wiki_json).transpose()

            wiki_data = wiki_df.drop_duplicates(['text']) # 3876
            wiki_data = wiki_data.reset_index(drop=True)

            # wiki_data['text'] = wiki_data['text'].apply(
            #     lambda x : ' '.join(re.sub(r'''[^ \r\nㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9ぁ-ゔァ-ヴー々〆〤一-龥~₩!@#$%^&*()“”‘’《》≪≫〈〉『』「」＜＞_+|{}:"<>?`\-=\\[\];',.\/·]''', 
            #     ' ', 
            #     str(x.lower().strip())).split())
            # )

            # wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\\n\\n',' '))
            # wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\n\n',' '))
            # wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\\n',' '))
            # wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\n',' '))

            self.contexts = [x for x in wiki_data['text']]
            print(f"context 1st example is {self.contexts[0]}")
            print(f"Lengths of unique contexts : {len(self.contexts)}")
        
        wiki_articles = [
            {"document_text": wiki_data['text'][i]} for i in range(len(wiki_data['text']))
        ]
        self.es = Elasticsearch("http://localhost:30001",timeout=30,max_retries=10,retry_on_timeout=True)
        print("Elasticsearch system setting")
        print('Elastic search ping:', self.es.ping())
        print('Elastic search info:')

        if self.es.indices.exists('document'):
            self.es.indices.delete(index='document')

        self.es.indices.create(index = 'document',
            body = {
                'settings':{
                    'analysis':{
                        'analyzer':{
                            'my_analyzer':{
                                "type": "custom",
                                'tokenizer':'nori_tokenizer',
                                'decompound_mode':'mixed',
                                'stopwords':'_korean_',
                                'synonyms':'_korean_',
                                "filter": ["lowercase",
                                            "my_shingle_f",
                                            "nori_readingform",
                                            "nori_number",
                                            "cjk_bigram",
                                            "decimal_digit",
                                            "stemmer",
                                            "trim"]
                            }
                        },
                        'filter':{
                            'my_shingle_f':{
                                "type": "shingle"
                            }
                        }
                    },
                    'similarity':{
                        'my_similarity':{
                            'type':'BM25',
                        }
                    }
                },
                'mappings':{
                    'properties':{
                        'text':{
                            'type':'text',
                            'analyzer':'my_analyzer',
                            'similarity':'my_similarity'
                        },
                    }
                }
            }
        )

        for i, text in enumerate(tqdm(wiki_articles)):
            try:
                self.es.index(index='document', id=i, body=text)
            except:
                print(f"Unable to load document {i}.")

    def retrieve(self,
                query_or_dataset: Union[str, Dataset],
                topk: Optional[int] = 1
                ) -> Union[Tuple[List, List], DatasetDict]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> DatasetDict: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        # tfidf 대신 es를 적용하는 것은 faiss와 별개이므로 faiss에 해당하는 부분은 건들지 않았습니다.
        if isinstance(query_or_dataset, str): # bulk = False
            if self.is_faiss:
                doc_scores, doc_indices = get_relevant_doc_faiss(query_or_dataset, self.tfidfv, self.indexer,  k=topk, bulk=False)
            elif not self.is_faiss:
                # doc_scores, doc_indices = get_relevant_doc_copy(query_or_dataset, self.tfidfv, self.passage_embedding,  k=topk, bulk=False)
                doc_scores, doc_indices = get_relevant_doc_es(query_or_dataset, self.es,k=topk,bulk=False)

            print("[Search query]\n", query_or_dataset, "\n")
            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset): # bulk = True
            if self.is_faiss:
                doc_scores, doc_indices = get_relevant_doc_faiss(query_or_dataset['question'], self.tfidfv, self.indexer, k=topk, bulk=True)
            elif not self.is_faiss:
                # doc_scores, doc_indices = get_relevant_doc_copy(query_or_dataset['question'], self.tfidfv, self.passage_embedding, k=topk, bulk=True)
                doc_scores, doc_indices = get_relevant_doc_es(query_or_dataset['question'], self.es, k=topk, bulk=True)

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    "question": example["question"], # query
                    "id": example["id"], # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            f = Features({"context": Value(dtype="string", id=None),
                          "id": Value(dtype="string", id=None),
                          "question": Value(dtype="string", id=None),})
            
            datasets = DatasetDict({"validation": Dataset.from_pandas(pd.DataFrame(total), features=f)})
            return datasets