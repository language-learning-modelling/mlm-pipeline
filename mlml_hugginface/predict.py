import json
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch.nn.functional import softmax
import time
import collections

class Predictor(object):
    '''
     Predicts a list of masked senteces
     for a given list of models
    '''
    def __init__(self, config_obj=None):
        self.config = config_obj
        self.masked_sent_tpl_lst_per_text = self.load_texts() 
        self.n_msk_sents = sum([len(l) for l in self.masked_sent_tpl_lst_per_text]) 
        print(f"processing a total of {len(self.masked_sent_tpl_lst_per_text)} texts and {self.n_msk_sents} masked sentences")

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.config.MODEL_CHECKPOINT
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_CHECKPOINT
        )
        self.vocab_size = len(self.tokenizer.vocab)
        self.top_k = self.config.TOP_K \
                     if self.config.TOP_K != "vocab"\
                     else self.vocab_size 
        model_num_parameters = self.model.num_parameters() / 1_000_000
        print(
            f"'>>> model number of parameters: {round(model_num_parameters)}M'"
        )


    def mlm_tpl(self,text_id, d):
        data = (
            (self.config.INPUT_FP,
            str(text_id),
            str(d['predictions']['maskedTokenIdx'])),
            d['predictions']['maskedSentence']['maskedSentenceStr']
            )
        return data

    def load_texts(self):
        return [[self.mlm_tpl(text_dict['text_id'], token_dict) 
                    for token_dict in text_dict["tokens"]]
                for text_dict in self.config.TEXTS.values()
                    
                ] 
        #raise Exception('Could not load texts')

    def check_if_skip_text(self, current_text):
        n_mlms = len(current_text)
        s=time.time()
        llm_tokens_first_msk_sent = self.tokenizer(
               current_text[0][1],
               padding=True,
               return_tensors='pt'
        )
        n_tokens = llm_tokens_first_msk_sent["input_ids"].size()[1]
        if n_tokens + 2 >= self.tokenizer.model_max_length:
            skip_text = True
            self.big_count+=1
            print(f"*"*100)
            print(f'{self.big_count} big sentences, this one has: {n_tokens} tokens')
            print(f"*"*100)
        else:
            skip_text = False
        return skip_text

    def batch(self):
        batch=[]
        self.big_count = 0
        while len(self.masked_sent_tpl_lst_per_text) > 0:
            masked_sent_tpl_lst = self.masked_sent_tpl_lst_per_text.pop()
            if self.check_if_skip_text(masked_sent_tpl_lst):
                continue

            while len(masked_sent_tpl_lst) > 0:
                masked_sent_tpl = masked_sent_tpl_lst.pop()
                batch.append(masked_sent_tpl)
                if len(batch) >= self.config.BATCH_SIZE:
                    yield batch
                    batch=[]
        if len(batch) > 0:
            yield batch


    def predict(self):
        batch_data = collections.defaultdict(lambda:[])
        with torch.no_grad():
            for batch_texts_tpls in self.batch():
                batch_texts_ids, batch_texts = zip(*batch_texts_tpls) 
                inputs = self.tokenizer(batch_texts,
                                   padding=True,
                                   return_tensors='pt')

                inputs.to(self.device)
                token_logits = self.model(**inputs).logits
                # Find the location of [MASK] and extract its logits
                mask_token_index = torch.where(
                    inputs['input_ids'] == self.tokenizer.mask_token_id
                )[1]
                mask_token_logits = token_logits[0, mask_token_index, :]
                mask_token_logits = softmax(mask_token_logits)
                mask_token_logits_ranked = torch.topk(
                            mask_token_logits,
                            self.top_k,
                            dim=1)
                token_idxs_per_instance = mask_token_logits_ranked.indices.tolist() 
                token_probs_per_instance = mask_token_logits_ranked.values.tolist()
                del token_logits
                del mask_token_index
                del mask_token_logits
                del mask_token_logits_ranked
                for instance_idx,(instance_tokens, instance_probs) in\
                        enumerate(zip(token_idxs_per_instance,
                            token_probs_per_instance)):  
                    for token_idx in range(len(instance_tokens)):
                        instance_dict = {
                                'token_vocab_idx':instance_tokens[token_idx],
                                'token_str':self.tokenizer.decode(
                                    [instance_tokens[token_idx]]
                                ),
                                'score': instance_probs[token_idx],
                        }
                        id_="_".join(batch_texts_ids[instance_idx])
                        batch_data[id_]\
                                .append(instance_dict)
                yield batch_data
                batch_data = collections.defaultdict(lambda:[])
            '''
            # Pick the [MASK] candidates with the highest logits
            print(dir(top_5_tokens[0]))
            print(f"{self.masked_texts}")
            for token_idx in top_5_tokens:
                print(
                    f"'>>> {self.masked_texts[0].replace(self.tokenizer.mask_token, self.tokenizer.decode([token_idx]))} {token_idx}'"
                )
            '''


if __name__ == '__main__':
    import sys
    config_filepath = sys.argv[1]
    print(config_filepath)
    predictor = Predictor(config_filepath)
    predictor.predict()
