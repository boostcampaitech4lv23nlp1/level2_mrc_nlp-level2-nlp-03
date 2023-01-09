import torch.nn as nn
import einops as ein

class LongformerModel(nn.Module):
    """_summary_
    베이스라인 모델입니다.
    """
    def __init__(self, model, num_labels, dropout_rate, num_layers=1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.model = model

        self.qa_outputs = nn.Sequential(
            # nn.Dropout(p=self.dropout_rate),
            nn.Linear(32000, self.num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = ein.rearrange(start_logits, 'batch seq 1 -> batch seq')
        end_logits = ein.rearrange(end_logits, 'batch seq 1 -> batch seq')

        return start_logits, end_logits