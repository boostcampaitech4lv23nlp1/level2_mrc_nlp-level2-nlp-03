import torch
import argparse
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import os
import numpy as np

from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk, load_metric, Dataset, DatasetDict, Features, Sequence, Value

import dataloader as DataProcess
import model as Model
from longformer import RobertaTokenizerFast, RobertaLongForMaskedLM

import utils.metric as Metric

import retrieval as Retrieval
from utils.seed_setting import seed_setting

def main(config):
    seed_setting(config.train.seed)
    assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
    device = torch.device('cuda')

    print('='*50,f'현재 적용되고 있는 전처리 클래스는 {config.data.preprocess}입니다.', '='*50, sep='\n\n')
    tokenizer = RobertaTokenizerFast.from_pretrained('klue_roberta_longformer', use_fast=True)
    prepare_features = getattr(DataProcess, config.data.preprocess)(tokenizer, config.train.max_length, config.train.stride)
    test_data = load_from_disk(config.data.test_path)

    retrieval = getattr(Retrieval, config.retrieval.retrieval_class)(
            tokenizer = tokenizer,
            data_path=config.retrieval.retrieval_path,
            context_path = config.retrieval.retrieval_data,
            is_faiss = config.retrieval.is_faiss
            )
    test_wiki_data = retrieval.retrieve(query_or_dataset=test_data, topk = config.retrieval.topk)['validation']

    test_dataset = test_wiki_data.map(
            prepare_features.test,
            batched=True,
            num_proc=1,
            remove_columns=test_wiki_data.column_names,
            load_from_cache_file=True,
            )

    metric = getattr(Metric, config.model.metric_class)(
                metric = load_metric('squad'),
                dataset = test_dataset,
                raw_data = test_wiki_data,
                n_best_size = config.train.n_best_size,
                max_answer_length = config.train.max_answer_length,
                save_dir = config.save_dir,
                mode = 'test'
            )

    test_dataset = test_dataset.remove_columns(["example_id", "offset_mapping"])
    test_dataset.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer)

    test_dataloader = DataLoader(test_dataset, batch_size= 8, collate_fn=data_collator, pin_memory=True, shuffle=False)
  
    print('='*50,f'현재 적용되고 있는 모델 클래스는 {config.model.model_class}입니다.', '='*50, sep='\n\n')
    model = getattr(Model, config.model.model_class)(
        model = RobertaLongForMaskedLM.from_pretrained('klue_roberta_longformer'),
        num_labels=2,
        dropout_rate = config.train.dropout_rate,
        )

    best_model = [model for model in os.listdir(f'./save/{config.save_dir}') if 'nbest' not in model and 'best' in model][0]
    checkpoint = torch.load(f'./save/{config.save_dir}/{best_model}')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # BERT일 경우 token_type_ids가 필요함
    is_token_type_ids = False
    check = True
    for model_name in ['roberta', 'distilbert', 'albert', 'camembert', 'flaubert']:
        if model_name in config.model.model_name:
            check = False
    if check and 'bert' in config.model.model_name:
        is_token_type_ids = True

    len_val_dataset = test_dataloader.dataset.num_rows
    start_logits_all, end_logits_all = [], []
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader):
            if is_token_type_ids: # BERT 모델일 경우 token_type_ids를 넣어줘야 합니다.
                inputs = {
                    'input_ids': test_batch['input_ids'].to(device),
                    'attention_mask': test_batch['attention_mask'].to(device),
                    'token_type_ids' : test_batch['token_type_ids'].to(device)
                }
            else:
                inputs = {
                        'input_ids': test_batch['input_ids'].to(device),
                        'attention_mask': test_batch['attention_mask'].to(device),
                    }

            start_logits, end_logits = model(**inputs)

            start_logits_all.append(start_logits.detach().cpu().numpy())
            end_logits_all.append(end_logits.detach().cpu().numpy())

    start_logits_all = np.concatenate(start_logits_all)[:len_val_dataset]
    end_logits_all = np.concatenate(end_logits_all)[:len_val_dataset]
    metrics = metric.compute_EM_f1(start_logits_all, end_logits_all, None)

if __name__=='__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline')
    args, _ = parser.parse_known_args()
    ## ex) python3 train.py --config baseline
    
    config = OmegaConf.load(f'./configs/{args.config}.yaml')
    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')

    main(config)