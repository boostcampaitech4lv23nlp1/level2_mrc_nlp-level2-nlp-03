import torch.nn as nn
import einops as ein
from transformers import T5ForConditionalGeneration, AutoModel

class BaselineModel(nn.Module):
    """_summary_
    베이스라인 모델입니다.
    """
    def __init__(self, model_name, num_labels, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.model_name = model_name
        
        if 't5' in self.model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).get_encoder()
        else:
            self.model = AutoModel.from_pretrained(self.model_name)
            
        self.qa_outputs = nn.Sequential(
            # nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.model.config.hidden_size, self.num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = ein.rearrange(start_logits, 'batch seq 1 -> batch seq')
        end_logits = ein.rearrange(end_logits, 'batch seq 1 -> batch seq')

        return start_logits, end_logits