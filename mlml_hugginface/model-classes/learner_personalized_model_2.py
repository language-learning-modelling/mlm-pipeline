from transformers import PreTrainedModel as HF_PreTrainedModel
from transformers import BertConfig, BertModel


class Model1(HF_PreTrainedModel):
    def __init__(self, config):
        super(NewModel, self).__init__(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(config.hidden_dropout_prob)
        self.model_name = config.model_name
        self.mlm_cls = nn.Linear(config.hidden_size, config.vocab_size)
        self.nationality_emb = nn.Embedding(
            n_of_nationalities, nationality_embedding_size
        )
        self.proficiency_emb = nn.Embedding(
            n_of_proficiencies, proficiency_embedding_size
        )
        self.user_emb = nn.Embedding(n_of_users, user_embedding_size)

    def forward(self, ids, mask, token_type_ids, natIdxs, profIDxs, userIdxs):
        n_embeds = self.nationality_emb(natIdxs).view((1,-1)) 
        p_embeds = self.proficiency_emb(profIdxs).view((1,-1)) 
        u_embeds = self.user_emb(userIdxs).view((1,-1)) 
        _, o2 = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        bo = self.bert_drop(o2)
        output = self.mlm_cls(bo)
        return output
