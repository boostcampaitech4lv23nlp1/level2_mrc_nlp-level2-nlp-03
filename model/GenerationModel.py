import torch.nn as nn
from transformers import T5ForConditionalGeneration

class GenerationModel(nn.Module):
    """_summary_
    생성모델입니다.
    """
    def __init__(self, model_name, num_labels, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.model_name = model_name
        if 't5' in self.model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        else: # T5가 아닌 경우, 구현이 필요하므로 에러를 발생시킵니다.
            raise NotImplementedError

    def forward(self, input_ids, attention_mask, labels = None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
