import torch
from transformers import BertTokenizer, BertModel

def add_tokens_to_bert_vocabulary(new_tokens_str, outputfp):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")

    print(len(tokenizer))  # 28996
    tokenizer.add_tokens(new_tokens_str)
    print(len(tokenizer))  # 28997

    model.resize_token_embeddings(len(tokenizer)) 
    # The new vector is added at the end of the embedding matrix

    print(model.embeddings.word_embeddings.weight[-1, :])
    # Randomly generated matrix

    model.save_pretrained(f'{outputfp}/model/')
    tokenizer.save_pretrained(f'{outputfp}/tokenizer/')
