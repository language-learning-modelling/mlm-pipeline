"""
train a pytorch model for the masked language modelling task using the transformers library
"""
import json
import random
import os
import time

import torch
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset as HF_Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from transformers import Trainer as HF_Trainer
from transformers import TrainingArguments as HF_TrainingArguments


class Trainer():
    def __init__(self, config_filepath):
        self.config = self.load_config(config_filepath)
        self.dataset = self.load_dataset(self.config['DATASET_NAME'])
        self.dataset_name = self.dataset_name()
        print(self.dataset_name)
        self.initial_model_name = self.get_initial_model_name_from_checkpoint()
        is_hf_checkpoint_given = self.config.get('HF_CHECKPOINT',False)
        if is_hf_checkpoint_given:
            self.model_folderpath = self.config['HF_CHECKPOINT'] 
            self.tokenizer_folderpath = self.config['HF_CHECKPOINT'] 
        else:
            self.expected_checkpoint_folder = f"./models/{self.initial_model_name}"
            self.expected_model_folder = f"{self.expected_checkpoint_folder}/model/"
            self.expected_tokenizer_folder = f"{self.expected_checkpoint_folder}/tokenizer/"
            self.model_folderpath = self.expected_model_folder  
            self.tokenizer_folderpath = self.expected_tokenizer_folder  
            
        print(self.config)
        self.tokenizer = self.load_tokenizer(folderpath=self.tokenizer_folderpath)
        self.model = self.load_model(folderpath=self.model_folderpath)

        print(f'>>> tokenizing dataset...')
        self.tokenized_dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns= list(self.dataset['train'][0])
        )
        print(f'>>> tokenized dataset: {self.tokenized_dataset["train"][0]}')

        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.config['MLM_PROBABILITY'],
            return_tensors='pt',
        )

        print(f'>>> Collating dataset...')
        """
        self.collated_dataset = self.tokenized_dataset.map(
            lambda dataset: self.mlm_collator(dataset['input_ids']),
            batched=True,
            batch_size=1000,
        )
        """

    def dataset_name(self):
        if '/' in self.config['DATASET_NAME']:
            is_filepath = True
        if is_filepath:
            last_char_is_slash = self.config['DATASET_NAME'][-1] == '/'
            if not last_char_is_slash:
                dataset_name = self.config['DATASET_NAME'].split('/')[-1]
            else:
                dataset_name = self.config['DATASET_NAME'].split('/')[-2]
        else:
            dataset_name = self.config['DATASET_NAME'] 
        return dataset_name 

    def get_initial_model_name_from_checkpoint(self):
        '''
            A model check point can either be a folderpath or a model checkpoint name
        '''
        if '/' in self.config['MODEL_CHECKPOINT']:
            is_filepath = True
        if is_filepath:
            last_char_is_slash = self.config['MODEL_CHECKPOINT'][-1] == '/'
            if not last_char_is_slash:
                initial_model_name = self.config['MODEL_CHECKPOINT'].split('/')[-1]
            else:
                initial_model_name = self.config['MODEL_CHECKPOINT'].split('/')[-2]
        else:
            initial_model_name = self.config['MODEL_CHECKPOINT']
        return initial_model_name 

    def load_tokenizer(self, folderpath):
        print(folderpath);input()
        tokenizer = AutoTokenizer.from_pretrained(
            folderpath, local_files_only=True
        )
        return tokenizer

    def load_model(self, folderpath):
        model = AutoModelForMaskedLM.from_pretrained(
            folderpath, local_files_only=True
        )
        return model

    def print_samples(self):
        tokenized_samples = [
            random.choice(self.tokenized_dataset['train']['input_ids'])
            for _ in range(10)
        ]
        for idx, sample in enumerate(tokenized_samples):
            print(
                f"'>>> Sampled Review {idx} length: {len(sample)}'", flush=True
            )
            print(f"'>>> Sampled Review {idx} length: {sample}'")

    def load_config(self, config_filepath):
        """
        params:
            config_filepath: str, e.g. "config.json"
        sets:
            config: dict
        examples:
            load_config("config.json")
            >>> {'MODEL_CHECKPOINT': 'bert-base-uncased', 'DATASET_NAME': 'imdb'}
        """
        with open(config_filepath) as inpf:
            config = json.load(inpf)
            config = {k.upper(): v for k, v in config.items()}
            return config

    def load_dataset(self, dataset_name):
        """
        params:
            dataset_name: str, e.g. "imdb"
        sets:
            dataset: datasets.DatasetDict
        examples:
            load_dataset("imdb")
            >>> DatasetDict({
                    train: Dataset({
                        features: ['label', 'text'],
                        num_rows: 25000
                    })
                    test: Dataset({
                        features: ['label', 'text'],
                        num_rows: 25000
                    })
                })
        """
        print(dataset_name)
        input()
        if os.path.isfile(dataset_name):
            # assuming is a .txt file
            # where each line is a unmasked sentence
            with open(dataset_name, encoding='utf-8') as inpf:
                texts = [
                    sent.replace('\n', '')
                    for sent in open(dataset_name, encoding='utf-8')
                ]
                my_dict = {'text': texts}
                dataset = HF_Dataset.from_dict(my_dict)
                dataset = dataset.train_test_split(
                    test_size=0.1, shuffle=True, seed=200
                )
        else:
            dataset = hf_load_dataset(dataset_name)
        sample = dataset['train'].shuffle(seed=42).select(range(3))
        for row in sample:
            print(f"\n'>>> {row['text']}'")
        return dataset

    def tokenize_function(self, example):
        """
        params:
            examples: dict
        sets:
            result: dict
        examples:
            tokenize_function({"text": "This is a sentence."})
            >>> {'attention_mask': [1, 1, 1, 1, 1, 1], 'input_ids': [101, 2023, 2003, 1037, 6251, 1012], 'token_type_ids': [0, 0, 0, 0, 0, 0]}
        """
        result = self.tokenizer(
            example['text'],
            add_special_tokens=True,
            return_special_tokens_mask=True,
            truncation=True,
        )
        if self.tokenizer.is_fast:
            result['word_ids'] = [
                result.word_ids(i) for i in range(len(result['input_ids']))
            ]

        return result

    def train(self):
        """
        params:
        sets:
        examples:
        """
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            # torch.cuda.current_device()
        print(torch.cuda.current_device())
        time.sleep(5)
        print(f'>>> Training model...')
        # Show the training loss with every epoch
        logging_steps = (
            len(self.tokenized_dataset['train']) // self.config['BATCH_SIZE']
        )
        model_name = self.config['MODEL_CHECKPOINT'].split('/')[-1]
        dataset_name = self.dataset_name
        finetunedModel_outputDir_fp=f'./models/{model_name}-finetuned-{dataset_name}'
        checkpoints_folderpath =f'{finetunedModel_outputDir_fp}/checkpoints' 
        # logging_dir=f'{finetunedModel_outputDir_fp}/logs'
        training_args = HF_TrainingArguments(
            output_dir=finetunedModel_outputDir_fp,
            # logging_dir="./logs/",#logging_dir,
            num_train_epochs= 10,
            overwrite_output_dir=True,
            logging_strategy='epoch',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=5e-5,
            weight_decay=0.01,
            per_device_train_batch_size=self.config['BATCH_SIZE'],
            per_device_eval_batch_size=self.config['BATCH_SIZE'],
            push_to_hub=False,
            fp16=False,
            # logging_steps=logging_steps,
            # save_steps=500,
            resume_from_checkpoint = True,
            save_total_limit=3,
            load_best_model_at_end=True
        )

        hf_trainer = HF_Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['test'],
            data_collator=self.mlm_collator,
            tokenizer=self.tokenizer,
        )
        hf_trainer.train()


if __name__ == '__main__':
    import sys
    config_filepath = sys.argv[1]
    print(config_filepath)
    trainer = Trainer(config_filepath)
    trainer.train()
