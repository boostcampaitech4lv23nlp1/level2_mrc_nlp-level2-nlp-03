import json
import pandas as pd

"""
AI hub의 '일반상식'의 WIKI 본문에 대한 질문-답 쌍 데이터를 추가하는 함수입니다.
"""

with open('data/ko_wiki_v1_squad.json', 'r') as f:
    data = json.load(f)

dict_data = {
    'id' : [],
    'title': [],
    'context':[],
    'question' : [],
    'answers' : [],
}
for row in data['data']:
    paragraph = row['paragraphs'][0]
    for col in paragraph['qas']:
        if col['answers'][0]['text'] in paragraph['context']:
            dict_data['id'].append(col['id'])
            dict_data['context'].append(paragraph['context'])
            dict_data['title'].append(row['title'])
            dict_data['question'].append(col['question'])
            dict_data['answers'].append(col['answers'][0])
df = pd.DataFrame(dict_data).sort_values(by='id')
df.to_csv('data/ko_wiki_v1_squad.csv', index=False)