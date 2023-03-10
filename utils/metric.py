import collections
from tqdm.auto import tqdm
import json
import numpy as np
from rouge_metric import PyRouge as Rouge
import more_itertools as mit
from collections import Counter

class Metrics_nbest():
    def __init__(self, metric, dataset, raw_data,  n_best_size, max_answer_length, save_dir, mode = 'train', version_2_with_negative=None, tokenizer=None):
        self.features = dataset # 원본 소스코드에서도 전처리가 완료된 dataset을 의미
        self.examples = raw_data # 원본 소스코드에서는 전처리 전의 원본 데이터인 example을 의미
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.metric = metric
        self.save_dir = save_dir
        self.mode = mode
        self.version_2_with_negative = version_2_with_negative
        self.tokenizer = tokenizer

    def compute_EM_f1(self, all_start_logits, all_end_logits, epoch=1):
        # 각 문서의 id를 키값, index를 밸류값으로 하는 딕셔너리 생성(참조용)
        '''
        키값으로 인덱스, 밸류값으로 동일한 아이디를 가지는 문서들의 index를 가지는 리스트(example_id_to_index 참조)
        ex) features_per_example[defaultdict] : {0: [0], 1: [1], 2: [2], 3: [3, 4], 4: [5, 6]}
        3 : [3,4]인 경우 document_id가 동일하지만 문장의 길이가 max_length보다 길어서 truncation되서 나눠진 데이터
        '''
        example_id_to_index = {k: i for i, k in enumerate(self.examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(self.features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)
        
        # prediction, nbest에 해당하는 OrderedDict 생성합니다.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        # if self.version_2_with_negative:
        #     scores_diff_json = collections.OrderedDict()

        # Prediction 시작(전체 example들에 대한 main Loop)
        for example_index, example in enumerate(tqdm(self.examples)):
            # 해당하는 현재 example_index(key) ex) 3 : [3,4]에서 feature_indices는 [3,4]에 해당됩니다.
            feature_indices = features_per_example[example_index]

            min_null_prediction = None # minimum null을 담을 공간을 초기화해줍니다.
            prelim_predictions = []
            
            # 현재 example에 대한 모든 feature 생성합니다.
            for feature_index in feature_indices: # ex) [3,4]
                # 각 featureure에 대한 모든 prediction을 가져옵니다.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]

                # logit과 original context의 logit을 mapping합니다.
                offset_mapping = self.features[feature_index]["offset_mapping"]

                # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
                token_is_max_context = self.features[feature_index].get(
                    "token_is_max_context", None # token_is_max_context가 없다면 None을 반환
                )

                # feature_null_score값이 이전에 truncation된 문장보다 작다면 현재 score값으로 업데이트 해줍니다.
                feature_null_score = start_logits[0] + end_logits[0] # sequence length 중에서 0번 인덱스는 cls vector를 의미
                if (min_null_prediction is None or min_null_prediction["score"] > feature_null_score):
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # n_best size만큼 큰 값 순으로 인덱스 정렬 및 reverse slicing([int:int:-1])
                start_indexes = np.argsort(start_logits)[-1 : (-self.n_best_size - 1) : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : (-self.n_best_size - 1) : -1].tolist()

                # n_best_size^2 만큼 완전탐색
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # out-of-scope answers는 고려하지 않습니다.
                        if (
                            start_index >= len(offset_mapping) # max_len을 벗어난 경우
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None # context가 아닌 question이나 special token일 경우
                            or offset_mapping[end_index] is None
                        ): continue

                        # 길이(end - start)가 < 0 또는 길이가 > max_answer_length(하이퍼 파라미터)인 answer도 고려하지 않습니다.
                        if (
                            end_index < start_index # 길이가 0 미만인 경우
                            or end_index - start_index + 1 > self.max_answer_length # max_answer_length보다 긴 경우
                        ): continue
                        
                        # 최대 context가 없는 answer도 고려하지 않습니다.
                        if (
                            token_is_max_context is not None # token_is_max_context이 None이라면 if문을 바로 빠져나옵니다.
                            and not token_is_max_context.get(str(start_index), False) # start_index가 포함되어 있지 않다면 반환
                        ): continue

                        # n_best_size내에서 고려할 수 있는 모든 경우를 추가합니다.
                        prelim_predictions.append(
                            {
                                "offsets": (
                                    offset_mapping[start_index][0],
                                    offset_mapping[end_index][1],
                                ),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
            # if version_2_with_negative:
            #     # minimum null prediction을 추가합니다.
            #     prelim_predictions.append(min_null_prediction)
            #     null_score = min_null_prediction["score"]

            # feature_indices(ex) [3,4])에 대한 탐색을 끝내고 모든 truncation 문장을 포함해서 가장 좋은 `n_best_size` predictions만 유지합니다.
            predictions = sorted(prelim_predictions,
                                 key=lambda x: x["score"], reverse=True # 내림차순
                                 )[:self.n_best_size] # n_best_size만큼 남기기
            
            # # 낮은 점수로 인해 제거된 경우 minimum null prediction을 다시 추가합니다.
            # if version_2_with_negative and not any(
            #     p["offsets"] == (0, 0) for p in predictions
            # ): predictions.append(min_null_prediction)

            # offset을 사용하여 original context에서 predict answer text를 수집합니다.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets") # offsets key의 value값을 pop  (start, end)
                pred["text"] = context[offsets[0] : offsets[1]] #predictions에 {'text' : predict answer text}를 추가

            # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                predictions.insert( # 예측에 실패했으므로 가장 높은 점수로써 기본 0값을 넣습니다.
                    0, {"text": "empty",
                        "start_logit": 0.0,
                        "end_logit": 0.0,
                        "score": 0.0}
                )
            # 모든 점수의 소프트맥스를 계산합니다
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # 예측값에 확률을 포함합니다.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # best prediction을 선택합니다. all_predictions에 id에 해당하는 가장 높은 확률[0]의 예상 text를 추가합니다.
            if not self.version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            # else:
                # # else case : 먼저 비어 있지 않은 최상의 예측을 찾아야 합니다
                # i = 0
                # while predictions[i]["text"] == "":
                #     i += 1
                # best_non_null_pred = predictions[i]

                # # threshold를 사용해서 null prediction을 비교합니다.
                # score_diff = (
                #     null_score
                #     - best_non_null_pred["start_logit"]
                #     - best_non_null_pred["end_logit"]
                # )
                # scores_diff_json[example["id"]] = float(score_diff)  # JSON-serializable 가능
                # if score_diff > null_score_diff_threshold:
                #     all_predictions[example["id"]] = ""
                # else:
                #     all_predictions[example["id"]] = best_non_null_pred["text"]

            # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
            all_nbest_json[example["id"]] = [
                { k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items() } for pred in predictions
            ]

        # all_predictions와 n_best를 json파일로 내보내기
        if self.mode == 'test':
            epoch = 'submission'
        with open(f'save/{self.save_dir}/predictions_{epoch}.json', "w", encoding="utf-8") as f:
            f.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + '\n')

        with open(f'save/{self.save_dir}/nbest_predictions_{epoch}.json', "w", encoding="utf-8") as f:
            f.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + '\n')

        # 실제 계산단계
        predicted_answers = [{"id": k, "prediction_text": v} for k, v in all_predictions.items()]
        if self.mode == 'train': # validation일때는 실제 값과 비교하여 계산
            theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.examples]
            return self.metric.compute(predictions=predicted_answers, references=theoretical_answers)
        else: # inference일 경우에는 예측값만 반환
            return predicted_answers
    def gen_compute_EM_f1(self, all_preds, epoch=1):
        # 각 문서의 id를 키값, index를 밸류값으로 하는 딕셔너리 생성(참조용)
        '''
        키값으로 인덱스, 밸류값으로 동일한 아이디를 가지는 문서들의 index를 가지는 리스트(example_id_to_index 참조)
        ex) features_per_example[defaultdict] : {0: [0], 1: [1], 2: [2], 3: [3, 4], 4: [5, 6]}
        3 : [3,4]인 경우 document_id가 동일하지만 문장의 길이가 max_length보다 길어서 truncation되서 나눠진 데이터
        '''
        # rouge = Rouge(rouge_n=2)
        if self.mode != "test":
            theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.examples]
        else:
            contexts = [{"id": ex["id"], "context": ex["context"]} for ex in self.examples]
        example_id_to_index = {k: i for i, k in enumerate(self.examples["id"])}
        features_per_example = collections.defaultdict(list)

        for i, feature in enumerate(self.features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)
        
        # prediction, nbest에 해당하는 OrderedDict 생성합니다.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()

        # Prediction 시작(전체 example들에 대한 main Loop)
        for example_index, example in enumerate(tqdm(self.examples)):
            # 해당하는 현재 example_index(key) ex) 3 : [3,4]에서 feature_indices는 [3,4]에 해당됩니다.
            feature_indices = features_per_example[example_index]
            preds = []
            unique_texts = set()

            # 현재 example에 대한 모든 feature 생성합니다.
            for feature_index in feature_indices: # ex) [3,4]
                # 생성된 텍스트를 전처리해줍니다.
                pred_text = post_process(all_preds[feature_index])
                pred=[{'prediction_text' : pred_text, 'id' : 1}]

                if self.mode != "test": # train or valid
                    answer = theoretical_answers[example_index]['answers']['text']
                    references=[{'answers' : {'answer_start': [0], 'text': [i]}, 'id' : 1} for i in answer]
                    score = self.metric.compute(predictions = pred, references = references)
                    em = score['exact_match']
                    f1 = score['f1']
                    # 기존 rouge score
                    # tok_pred = self.tokenizer.tokenize(pred_text)
                    # tok_answer = [self.tokenizer.tokenize(text) for text in answer]
                    # score = 0
                    # rouge_score = rouge.evaluate_tokenized([tok_pred], tok_answer)['rouge-2']['f']
                    # score = max(score, rouge_score)
                    preds.append(
                    {
                        "text" : pred_text,
                        "real_ans" : answer,
                        "em" : em,
                        "f1" : f1,
                        # "score" : score
                    }
                    )
                else:
                    """
                    test인 경우, 비교할 정답이 없으므로 다른 방식으로 순위를 매겨야 합니다.
                    정답이 있는 경우, 지문 내 항상 존재하므로 지문에 생성된 정답이 존재하는지 파악하고, 그 뒤 지문에 많이 등장하는 순서로 정렬하였습니다.
                    정답이 여러번 생성된 경우, 가중치를 주기 위해 점수를 더해주었습니다.
                    """
                    other_score = get_score(self.tokenizer.encode(pred_text), self.tokenizer.encode(contexts[example_index]['context']))

                    if pred_text in unique_texts:
                        # 이미 등록된 정답인 경우, 점수를 더해줍니다.
                        index = list(mit.locate(preds, pred=lambda x: x["text"] == pred_text))
                        preds[index[0]]["other_score"] += other_score
                    else:
                        # 등록되지 않은 정답인 경우, 리스트에 등록해줍니다.
                        if pred_text in contexts[example_index]['context']:
                            preds.append(
                                {
                                    "text" : pred_text,
                                    "em" : 1,
                                    "other_score" : other_score
                                }
                            )
                        else:
                            preds.append({
                                "text" : pred_text,
                                "em" : 0,
                                "other_score" : other_score
                            })
                        unique_texts.add(pred_text)
            if self.mode != "test":
                predictions = sorted(preds,
                            key=lambda x: (x["em"], x['f1']), reverse=True # 내림차순
                            )[:self.n_best_size] # n_best_size만큼 남기기
            else:
                predictions = sorted(preds,
                            key=lambda x: (x["em"], x['other_score']), reverse=True # 내림차순
                            )[:self.n_best_size] # n_best_size만큼 남기기

            # best prediction을 선택합니다. all_predictions에 id에 해당하는 가장 높은 확률[0]의 예상 text를 추가합니다.
            if not self.version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            all_nbest_json[example["id"]] = [
                { k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items() } for pred in predictions
            ]
                
        # all_predictions와 n_best를 json파일로 내보내기
        if self.mode == 'test':
            epoch = 'submission'
        with open(f'save/{self.save_dir}/predictions_{epoch}.json', "w", encoding="utf-8") as f:
            f.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + '\n')
        with open(f'save/{self.save_dir}/nbest_predictions_{epoch}.json', "w", encoding="utf-8") as f:
            f.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + '\n')

        # 실제 계산단계
        predicted_answers = [{"id": k, "prediction_text": v} for k, v in all_predictions.items()]
        if self.mode == 'train': # validation일때는 실제 값과 비교하여 계산
            return self.metric.compute(predictions=predicted_answers, references=theoretical_answers)
        else: # inference일 경우에는 예측값만 반환
            return predicted_answers

def post_process(string):
    return (string.split("</s>")[0].replace("<pad>", "").strip())

def get_score(word, context):
    counted = Counter(context)
    val = 0
    for i in word:
        val += counted[i]
    return val/sum(counted.values())