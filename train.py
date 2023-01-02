import torch
import argparse
from omegaconf import OmegaConf

from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk, load_metric
from sklearn.model_selection import StratifiedKFold

import dataloader as DataProcess
import trainer as Trainer
import model as Model

import torch.optim as optim
import utils.loss as Criterion
import utils.metric as Metric

from utils.check_dir import check_dir
from utils.wandb_setting import wandb_setting
from utils.seed_setting import seed_setting
from utils.AIhub_data_add import AIhub_data_add


def main(config):
    seed_setting(config.train.seed)
    assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
    device = torch.device('cuda')
    
    print('='*50,f'현재 적용되고 있는 전처리 클래스는 {config.data.preprocess}입니다.', '='*50, sep='\n\n')
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, use_fast=True)
    prepare_features = getattr(DataProcess, config.data.preprocess)(tokenizer, config.train.max_length, config.train.stride)

    # data Augementation
    if config.data.get('AIhub_data_add'):
        train_data = AIhub_data_add(config.data.train_path)
    else:
        train_data = load_from_disk(config.data.train_path)
    valid_data = load_from_disk(config.data.val_path)
    
    # 데이터셋 로드 클래스를 불러옵니다.
    train_dataset = train_data.map(
            prepare_features.train,
            batched=True,
            num_proc=4,
            remove_columns=train_data.column_names,
            load_from_cache_file=True,
        )
    valid_dataset = valid_data.map(
            prepare_features.valid,
            batched=True,
            num_proc=4,
            remove_columns=valid_data.column_names,
            load_from_cache_file=True,
        )

    # 원본 test data와 test dataset을 넣어주셔야 합니다.
    metric = getattr(Metric, config.model.metric_class)(
                metric = load_metric('squad'),
                dataset = valid_dataset,
                raw_data = valid_data,
                n_best_size = config.train.n_best_size,
                max_answer_length = config.train.max_answer_length,
                save_dir = config.save_dir,
                mode = 'train'
            )

    train_dataset.set_format("torch")
    valid_dataset = valid_dataset.remove_columns(["example_id", "offset_mapping"])
    valid_dataset.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size= config.train.batch_size, collate_fn=data_collator, pin_memory=True, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size= config.train.batch_size, collate_fn=data_collator, pin_memory=True, shuffle=False)

    # 모델 아키텍처를 불러옵니다.
    print('='*50,f'현재 적용되고 있는 모델 클래스는 {config.model.model_class}입니다.', '='*50, sep='\n\n')
    model = getattr(Model, config.model.model_class)(
        model_name = config.model.model_name,
        num_labels=2,
        dropout_rate = config.train.dropout_rate,
        ).to(device)

    criterion = getattr(Criterion, config.model.loss)
    optimizer = getattr(optim, config.model.optimizer)(model.parameters(), lr=config.train.learning_rate)
    
    lr_scheduler = None
    epochs = config.train.max_epoch
    save_dir = check_dir(config.save_dir)

    print('='*50,f'현재 적용되고 있는 트레이너는 {config.model.trainer_class}입니다.', '='*50, sep='\n\n')
    trainer = getattr(Trainer, config.model.trainer_class)(
            model = model,
            criterion = criterion,
            metric = metric,
            optimizer = optimizer,
            device = device,
            save_dir = save_dir,
            train_dataloader = train_dataloader,
            valid_dataloader = valid_dataloader,
            lr_scheduler=lr_scheduler,
            epochs=epochs,
        )

    ## wandb를 설정해주시면 됩니다. 만약 sweep을 진행하고 싶다면 sweep=True로 설정해주세요.
    ## 자세한 sweep 설정은 utils/wandb_setting.py를 수정해주세요.
    wandb_setting(config)
    trainer.train()

if __name__=='__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline')
    args, _ = parser.parse_known_args()
    ## ex) python3 train.py --config baseline
    
    config = OmegaConf.load(f'./configs/{args.config}.yaml')
    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')

    main(config)