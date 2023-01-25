<img src="https://user-images.githubusercontent.com/113564875/214513091-a3cd1bfc-2e83-4bf4-b9c9-7df1b25421b3.png">

## Contents
1. [Overview](#1.-overview)
2. [Project Tree](#2-project-tree)
3. [Solutions](#3-solutions)
4. [Model Config](#4-model-config)
5. [Contributors](#5-contributors)
6. [Git Convention](#6-commit-convention)

<br></br>

## 1. Overview
### 소개
- Question Answering (QA)은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다.
- 다양한 QA 시스템 중, Open-Domain Question Answering (ODQA) 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가되기 때문에 더 어려운 문제입니다.

### Metric
1. Exact Match (EM): 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수가 주어집니다. 즉 모든 질문은 0점 아니면 1점으로 처리됩니다. 
2. F1 Score: EM과 다르게 부분 점수를 제공합니다.
<br></br>

## 2. Project Tree
```bash.  
|-- configs        # 모델 별 config 저장
|-- data           # train,test,wiki 데이터
|-- dataloader              
|-- elasticsearch-7.15.1    # elastic search가 설치된 경로
|-- model                   
|-- retrieval               
|-- save           # 모델 pt 저장
|-- test           # 실험 관련 파일들 저장
|-- trainer                 
|-- utils          # 데이터 증강, 디렉토리 존재 유무 확인 함수 등
`-- wandb  
```
<br></br>

## 3. Solutions
## Retriever
### Sparse Retriever
- TF-IDF : 베이스라인에서 사용된 방식입니다. top-k를 일정 수준까지 상승시키는 것이 성능 향상으로 이어졌습니다. 최적의 top-k는 40이었습니다. 
- BM25 : TF-IDF를 개선하여 전체 문서의 평균 길이를 반영하고 TF의 영향력을 낮춰줍니다. 기존 TF-IDF에 비해 우수한 성능을 보여주었으며, 최적의 top-k는 30이었습니다.
<img src="https://user-images.githubusercontent.com/113564875/214514391-f7d33950-8967-4771-ae06-610eead1978e.png">

- Elastic Search : Apache Lucene 기반의 Java 오픈소스 분산 검색 엔진입니다. 문서를 탐색하는 근거가 BM25인데 성능이 하락하는 결과가 나타났습니다. 그 이유로는 부적절한 tokenizer의 사용을 들 수 있습니다.  

### Dense Retriever
- ODQA라는 확장된 개념에서 기존의 문제점을 고려하여 context에 없는 답을 도출하는 상황을 연출하기 위해 Dense Retriever를 실험해보았습니다.
-  In batch negative : 관련 있는 passage는 벡터 표현상 가깝게, 관련 없는 passage는 멀게 밀어낸다는 개념입니다.
- Sparse Retriever보다 낮은 성능 : 본 대회에서는 context 내의 모든 의미를 학습하는 것보다 데이터의 구조를 학습하는 것이 더 유의미한 학습 방식이었습니다.
<br></br>

## Reader
저희는 hyper parameter tuning을 초반부터 진행하여 대회에 가장 적합하다고 생각하는 모델을 빨리 파악하고자 했습니다.
- klue/bert-base : 베이스라인에서 사용된 모델입니다.
- klue/roberta-large : 저희는 본 대회의 task가 특정 span을 기반으로 하는 RE task와 유사하다고 판단했습니다. 결과적으로 가장 우수한 성능을 보인 모델입니다.

### Extraction Model
1. Parameter size
- paust-T5 : parameter size가 360M으로 klue/
    roberta-large보다 큰 모델이지만 성능은 오히려 하락했습니다.
- LSTM/GRU layer : 모델의 parameter size를 키움과 동시에 span의 순서 정보를 추가 학습시킴으로써 성능 향상을 꾀했습니다. 그러나 오히려 성능이 떨어지는 경우도 존재했습니다.
- 본 대회에서는 단순히 parameter size를 키우는 것보다 어떻게 span을 잘 추출할 수 있도록 학습시키는지가 더욱 중요하다고 판단했습니다.
2. More Longer Max Length
- baseline의 max_length: 384 설정을  max_length: 512로 설정할 경우 성능이 더 올라가는 것을 확인하였습니다.
- monologg/kobigbird-bert-base : max_length: 4096를 볼 수 있는 모델입니다. 하지만 오히려 성능이 하락하는 결과가 나타났습니다.
<img width="1066" alt="kobigbird" src="https://user-images.githubusercontent.com/113564875/214512635-c731d6b1-a96b-4b08-aaea-e337c48e24b1.png">

- Longformer : sparse attention과 global attention을 적용하여 4096개의 토큰을 학습할 수 있는 모델입니다. klue/roberta 모델의 attention을 Longformer의 attention으로 교체한 후, wiki data에 대해 MLM task를 pretraining 작업을 진행하였습니다. 마찬가지로 성능이 하락하는 결과가 나타났습니다.
<img width="988" alt="longformer" src="https://user-images.githubusercontent.com/113564875/214512859-edd9ca6d-9604-4939-80cf-9d287a474903.png">

- 모델이 문장의 손실 없이 학습을 하게 되면 성능이 향상될 것이라는 기대와 다른 결과가 나타났습니다. 저희는 이것이 모델이 긴 문장을 받아들이며 정답을 추출하는데 불필요한 부분들을 참고하게 되어 혼란이 가중되었기 때문이라고 추측했습니다.

3. Relative Position
- lighthouse/mdeberta-v3-base-kor-further: 각 토큰 간의 상대적 거리를 학습한 모델이 SQuAD 데이터셋에서 좋은 성능을 보인다는 것을 근거로 KorQuAD에서도 각 토큰 간의 상대적 거리를 학습한다면 더 좋은 성능을 보일 것이라 생각하였습니다.
- DeBERTa는 한국어를 처리하기에 부적합한 모델이므로 KPMG에서 한국어 데이터를 추가적으로 학습한 위 모델을 사용하여 fine-tuning을 진행했습니다. 

<img width="1092" alt="deberta" src="https://user-images.githubusercontent.com/113564875/214513371-2563f9a7-9da6-467b-8efb-e74ea09473d0.png">

- 여기서도 다소 아쉬운 결과가 나타난 원인으로는 tokenizer를 꼽았습니다. morpheme-based wordpiece를 사용하는 klue/roberta와 달리 DeBERTa 모델은 sentence piece를 사용했기 때문이라는 것입니다.

### Generation Model
본 대회의 데이터셋은 정답이 지문에 포함된 데이터셋이므로, 지문에서 정답의 위치를 추출하는 모델보다 생성 모델이 성능이 떨어질 것이라는 가설을 세웠습니다. 실제 실험을 통해 확인한 결과는 다음과 같습니다.
- KE-T5 : 학습을 진행하면 할수록 한 단어만 계속 반복하는 문제가 발생하여 추가 실험을 진행하지 않았습니다.
- paust/pko-t5-large : 훈련 시 EM Score가 50점 정도를 보였고, epoch를 늘려 훈련한 결과 64점까지 나왔지만, test 데이터를 사용하여 생성했을 때는 16점으로 매우 낮은 수치를 보여주었습니다.
- 모델이 생성한 여러 정답들 중 질문과 가장 연관이 있는 단어를 선택하기 위한 정렬 알고리즘이 필요한데, 적절한 알고리즘을 구현하지 못한 것이 훈련과 테스트 간의 점수 차이인 것으로 추측했습니다. 
- 또한 top-k가 늘어나는 경우 모델이 봐야 할 문서가 늘어나고, 이에 따라 생성되는 정답의 수 또한 늘어나게 되는데, 이 정답들의 우선순위를 제대로 매기지 못하여 오답이 선택될 가능성이 상승한 것으로 판단했습니다.


## 4. Model Config
Roberta-large

- Batch_size : 8
- Learning_rate : 5e-6
- epochs : 2
- optimizer : AdamW
- loss function : Cross Entropy

## 5. Contributors
|김근형|김찬|유선종|이헌득|
|:---:|:---:|:---:|:---:|
|<img src="https://user-images.githubusercontent.com/97590480/205299519-174ef1be-eed6-4752-9f3d-49b64de78bec.png">|<img src="https://user-images.githubusercontent.com/97590480/205299316-ea3dc16c-00ec-4c37-b801-3a75ae6f4ca2.png">|<img src="https://user-images.githubusercontent.com/97590480/205299037-aec039ea-f8d3-46c6-8c11-08c4c88e4c56.jpeg">|<img src="https://user-images.githubusercontent.com/97590480/205299457-5292caeb-22eb-49d2-a52e-6e69da593d6f.jpeg">|
|[Github](https://github.com/kimkeunhyeong)|[Github](https://github.com/chanmuzi)|[Github](https://github.com/Trailblazer-Yoo)|[Github](https://github.com/hundredeuk2)|
- 김근형[Reader] : Generation Model 구현
- 김찬[Retriever] : BM25, Elasticsearch 구현
- 유선종[Reader] : roberta-large, DeBerta, Longformer 구현
- 이헌득[Retriever]  : Dense Retriever 구현
<br></br>

## 6. Commit Convention.

**커밋 메세지 스타일 가이드**

### Commit Message Structure.

기본적으로 커밋 메세지는 아래와 같이 구성한다.

```bash
type: subject (#이슈번호)
```

**Git commit message 참고할 블로그.**

[좋은 git commit 메시지를 위한 영어 사전](https://blog.ull.im/engineering/2019/03/10/logs-on-git.html)

### Commit type

- `feat`: 새로운 기능 추가
- `fix`: 버그 수정, Simplify
- `docs`: 문서 수정, 주석 수정
- `delete`: 삭제(remove) 
- `style`: 코드 포맷팅, 세미콜론 누락, 코드 변경이 없는 경우
- `refactor`: 코드 리펙토링
- `test`: 테스트 코드, 리펙토링 테스트 코드 추가

### Subject

제목은 50자를 넘기지 않고, 첫 명령어에만 대문자로 작성, 마지막에 마침표를 붙이지 않는다.

과거시제를 사용하지 않고 명령어로 작성한다.

- “Fixed” → “Fix”
- “Added” → “Add”

------

## PR convention.

```bash
## 개요
`어떤 이유에서 이 PR을 시작하게 됐는지에 대한 히스토리를 남겨주세요.`

## 작업사항
`해당 이슈사항을 해결하기 위해 어떤 작업을 했는지 남겨주세요.`

## 로직
`어떤 목적을 가지고 딥러닝 코드를 작성했는지 간략히 써주세요.`

## Resolved
`해결한 Issue 번호를 적어주세요.`
```

## issue convention.

```bash
## 목표
`어떤 목표를 가지고 작업을 진행하는지 남겨주세요.`

## 세부사항
`어떤 세부사항이 예상되는지 작성해주세요.`        
or
## Target 
`어떤 부분을 수정하거나 추가가 할 것인지 작성해주세요.`        

## 세부사항
`어떤 세부사항을 수정/추가할 것인지 작성해주세요.`       
```

## Discussions convention.

##### Discussion Message Structure.

기본적으로 Discussion은 다음을 포함하여 게시한다.

```bash
type: Title (#이슈번호)

Current Status 

Analysis Result

Conculsion

Disscusion about ? (Optional)
```

##### Commit type

- `Ideas`: 논문 리스트, 방향 제시, 아이디어
- `Polls`: 수정 제안, ex) Add convention rule 
- `Q&A`: 질의
- `Show & Tell`: 결과 분석 결과 공유
- `General`: 그 외 
