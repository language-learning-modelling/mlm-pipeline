import json
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig
import torch
from torch.nn.functional import softmax
import time
import collections

from transformers import AutoModelForMaskedLM, AutoTokenizer
import os

class ModelLoader:
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.model = None
        self.tokenizer = None

    def load_checkpoint(self):
        # Check if both model and tokenizer files are in the same directory
        if self._has_model_and_tokenizer_files(self.model_checkpoint):
            # Case 1: Load directly from the checkpoint directory
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        else:
            # Case 2: Check for model/ and tokenizer/ subdirectories
            model_path = os.path.join(self.model_checkpoint, 'model')
            tokenizer_path = os.path.join(self.model_checkpoint, 'tokenizer')
            
            if os.path.exists(model_path) and os.path.exists(tokenizer_path):
                # Load model and tokenizer from subdirectories
                self.model = AutoModelForMaskedLM.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                raise ValueError(f"Checkpoint at {self.model_checkpoint} is not properly structured.")

    def _has_model_and_tokenizer_files(self, directory):
        # Check if directory contains files necessary for both model and tokenizer
        model_files = ['generation_config.json', 'config.json', 'model.safetensors']  # Common model files
        tokenizer_files = ['vocab.txt', 'tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json']  # Common tokenizer files

        model_files_present = all(os.path.isfile(os.path.join(directory, f)) for f in model_files)
        tokenizer_files_present = all(os.path.isfile(os.path.join(directory, f)) for f in tokenizer_files)

        return model_files_present and tokenizer_files_present

class Predictor(object):
    '''
     Predicts a list of masked senteces
     for a given list of models
    '''
    def __init__(self, config_obj=None):
        self.config = config_obj
        self.big_count = 0
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model_loader = ModelLoader(self.config.MODEL_CHECKPOINT)
        self.model_loader.load_checkpoint()
        self.model = self.model_loader.model 
        self.model.to(self.device)
        self.tokenizer = self.model_loader.tokenizer 
        self.vocab_size = len(self.tokenizer.vocab)

        self.masked_sent_tpl_lst_per_text = self.load_texts() 
        self.n_msk_sents = sum([len(l) for l in self.masked_sent_tpl_lst_per_text]) 
        print(f"processing a total of {len(self.masked_sent_tpl_lst_per_text)} texts and {self.n_msk_sents} masked sentences")
        self.top_k = self.config.TOP_K \
                     if self.config.TOP_K != "vocab"\
                     else self.vocab_size 
        model_num_parameters = self.model.num_parameters() / 1_000_000
        print(
            f"'>>> model number of parameters: {round(model_num_parameters)}M'"
        )


    def generate_masked_sentences(self, text_dict):
        masked_sentences = []
        tokens = [d['token']['token_str'] for d in text_dict["tokens"]]
        for token_idx, _ in enumerate(tokens):
            masked_token_sentence = tokens.copy()
            masked_token_sentence[token_idx] = self.tokenizer.mask_token  
            masked_sentences.append(
                    {
                        "idx":f"{text_dict['text_id']}_{token_idx}",
                        "text":" ".join(masked_token_sentence)
                    }
            )
        return masked_sentences

    def mlm_tpl(self,text_id, d, msk_sent_str):
        data = (
            (self.config.INPUT_FP,
            str(text_id),
            str(d['predictions']['maskedTokenIdx'])),
            msk_sent_str
            )
        return data

    def load_texts(self):
        texts=[]
        for text_dict in self.config.TEXTS.values():
            text_mlms=[]
            if self.check_if_skip_text(text_dict):
                continue
            for token_dict, masked_sentence_dict\
                    in zip(
                            text_dict["tokens"],
                            self.generate_masked_sentences(text_dict)
                            ):
                        text_mlms.append(self.mlm_tpl(
                            text_dict['text_id'],
                            token_dict,
                            masked_sentence_dict['text']
                        )) 
            texts.append(text_mlms)
        return texts
        #raise Exception('Could not load texts')

    def increment_big_count(self, v, n_tokens):
        self.big_count+=n_tokens
        print(f"*"*100)
        print(f'{self.big_count} masked tokens were skipped')
        print(f"*"*100)
        return v

    def check_if_skip_text(self, text_dict):
        llm_tokens_first_msk_sent = self.tokenizer(
                text_dict['text'],
                padding=True,
                return_tensors='pt'
        )
        n_tokens = llm_tokens_first_msk_sent["input_ids"].size()[1]
        # print(n_tokens, self.tokenizer.model_max_length);input()
        skip_text = self.increment_big_count(True, n_tokens) if n_tokens > self.tokenizer.model_max_length else False
        return skip_text

    def batch(self):
        batch=[]
        while len(self.masked_sent_tpl_lst_per_text) > 0:
            masked_sent_tpl_lst = self.masked_sent_tpl_lst_per_text.pop()
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
                try:
                    token_logits = self.model(**inputs).logits
                except:
                    print(f"{batch_texts_ids}",file=open("error_pred.txt","a"))
                    continue
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
