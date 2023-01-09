from tqdm.auto import tqdm
import torch
import wandb
import numpy as np
import os
import time
import gc

class GenerationTrainer():
    """
    훈련과정입니다.
    """
    def __init__(self, model, criterion, metric, optimizer, device, save_dir,
                 train_dataloader, valid_dataloader=None, lr_scheduler=None, epochs=1, tokenizer=None, max_answer_length = None):
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.tokenizer = tokenizer
        self.max_answer_length = max_answer_length

        self.best_model_epoch, self.val_loss_values = [], []

    def train(self):
        """
        train_epoch를 돌고 valid_epoch로 평가합니다.
        """
        for epoch in range(self.epochs):
            standard_time = time.time()
            self._train_epoch(epoch)
            self._valid_epoch(epoch)
            wandb.log({'epoch' : epoch, 'runtime(Min)' : (time.time() - standard_time) / 60})
        self.select_best_model()
        torch.cuda.empty_cache()
        del self.model, self.train_dataloader, self.valid_dataloader
        gc.collect()
    
    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        epoch_loss = 0
        steps = 0
        pbar = tqdm(self.train_dataloader)
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            steps += 1
            inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'labels' : batch['labels'].to(self.device)
            }
            outputs = self.model(**inputs)

            loss = outputs.loss
            loss.backward()

            epoch_loss += loss.detach().cpu().numpy().item()
            
            self.optimizer.step()
            
            pbar.set_postfix({
                'loss' : epoch_loss / steps,
                'lr' : self.optimizer.param_groups[0]['lr'],
            })
            wandb.log({'train_loss':epoch_loss/steps})
        pbar.close()

    def _valid_epoch(self, epoch):
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            self.model.eval()
            all_preds = []
            for valid_batch in tqdm(self.valid_dataloader):
                val_steps += 1
                inputs = {
                        'input_ids': valid_batch['input_ids'].to(self.device),
                        'attention_mask': valid_batch['attention_mask'].to(self.device),
                        'labels' : valid_batch['labels'].to(self.device)
                }

                outputs = self.model(**inputs)

                loss = outputs.loss
                val_loss += loss.detach().cpu().numpy().item()

                # 답변을 생성해 줍니다.
                pred_ids = self.model.model.generate(
                    input_ids = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    max_length = self.max_answer_length,
                    do_sample=True,
                    top_p=0.95, 
                    top_k=50
                )

                pred_ids = pred_ids.cpu().numpy()
                
                # 생성된 답변을 디코딩해 문자열로 저장해줍니다.
                for pred_id in pred_ids:
                    pred_decoded = self.tokenizer.decode(pred_id)
                    all_preds.append(pred_decoded)
                    
            val_loss /= val_steps

            metrics = self.metric.gen_compute_EM_f1(all_preds, epoch)
            print(f"Epoch [{epoch+1}/{self.epochs}] Val_loss : {val_loss}")
            print(f"Epoch [{epoch+1}/{self.epochs}] Extact Match :", metrics['exact_match'])
            print(f"Epoch [{epoch+1}/{self.epochs}] F1_score :", metrics['f1'])
            wandb.log({'epoch' : epoch+1, 'val_loss' : val_loss})
            wandb.log({'epoch' : epoch+1, 'Exact_Match' : metrics['exact_match']})
            wandb.log({'epoch' : epoch+1, 'f1_score' : metrics['f1']})

            if epoch < 9:
                epoch = '0' + str(epoch+1)
            else:
                epoch = epoch + 1
            torch.save(self.model.state_dict(), f'save/{self.save_dir}/epoch:{epoch}_model.pt')
            print('save checkpoint!')

        self.best_model_epoch.append(f'save/{self.save_dir}/epoch:{epoch}_model.pt')
        self.val_loss_values.append(val_loss)

    def select_best_model(self):
        best_model = self.best_model_epoch[np.array(self.val_loss_values).argmin()]
        os.rename(best_model, best_model.split('.pt')[0] + '_best.pt')