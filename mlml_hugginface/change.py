import torch
import torch.nn as nn
from transformers import PreTrainedModel as HF_PreTrainedModel
from transformers import BertConfig, BertModel


class NewModel(HF_PreTrainedModel):
    def __init__(self, config):
        super(NewModel, self).__init__(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(config.hidden_dropout_prob)
        self.out = nn.Linear(config.hidden_size, config.vocab_size)
        self.model_name = config.model_name

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output


if __name__ == '__main__':
    config = BertConfig(model_name='bert_model')
    model = NewModel(config)
