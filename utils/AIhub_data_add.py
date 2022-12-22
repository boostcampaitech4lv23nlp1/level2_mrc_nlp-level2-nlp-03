from datasets import Dataset,load_from_disk, concatenate_datasets
from ast import literal_eval
import pandas as pd

def AIhub_data_add(train_path:str):
    dataset1 = load_from_disk(train_path)
    dataset1 = dataset1.remove_columns(['document_id', '__index_level_0__'])

    df = pd.read_csv('data/ko_wiki_v1_squad.csv')
    df['answers'] = df['answers'].apply(lambda x: literal_eval(x))
    df['answers'] = df['answers'].apply(lambda x: {'answer_start' : [int(x['answer_start'])], 'text': [x['text']]})

    dataset2 = Dataset.from_pandas(df)
    print('='*3, 'AI hub 데이터 추가 완료', '='*3)

    return concatenate_datasets([dataset1, dataset2])