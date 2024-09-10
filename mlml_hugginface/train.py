import json
import random
import os
import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import srsly 
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

from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# Define the enum for training strategies
class TrainingStrategy(Enum):
    FULL_LLM_TOKENIZE = "FULL+LLM-TOKENIZE"
    RESUME_LLM_TOKENIZE = "RESUME+LLM-TOKENIZE"
    FULL_HUMAN_TOKENIZE = "FULL+HUMAN-TOKENIZE"

# Reverse mapping from string to enum
# Automatically generate reverse mapping from the enum values
TRAINING_STRATEGY_MAP = {strategy.value: strategy for strategy in TrainingStrategy}

@dataclass
class TrainerConfig:
    MODEL_CHECKPOINT: str
    DATASET_NAME: str
    DATASET_FOLDER: str = "datasets"
    SPLIT: str = None
    HF_CHECKPOINT: bool = False
    LORA: bool = False
    MLM_PROBABILITY: float = 0.15
    BATCH_SIZE: int = 16
    # Allow training_strategy as a string input, which will be converted to enum
    TRAINING_STRATEGY: str = field(default="FULL+LLM-TOKENIZE")

    def __post_init__(self):
        required_fields = ["MODEL_CHECKPOINT", "DATASET_NAME"]
        for field_key in self.__dataclass_fields__.keys():
            if field_key in required_fields and self.__getattribute__(field_key) is None:
                raise ValueError(f'missing {field_key} config property')

        # Convert the string training_strategy to enum if it's a valid string
        if isinstance(self.TRAINING_STRATEGY, str):
            if self.TRAINING_STRATEGY not in TRAINING_STRATEGY_MAP:
                raise ValueError(f'Invalid training strategy: {self.TRAINING_STRATEGY}')
            self.TRAINING_STRATEGY = TRAINING_STRATEGY_MAP[self.TRAINING_STRATEGY]
        elif not isinstance(self.TRAINING_STRATEGY, TrainingStrategy):
            raise ValueError(f'Invalid training strategy type: {self.TRAINING_STRATEGY}')


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.dataset_name = self.dataset_name()
        self.dataset = self.load_dataset(self.config.DATASET_NAME)
        exit()
        self.initial_model_name = self.get_initial_model_name_from_checkpoint()

        # self.save_data_splits()

        if self.config.HF_CHECKPOINT:
            self.model_folderpath = self.config.HF_CHECKPOINT
            self.tokenizer_folderpath = self.config.HF_CHECKPOINT
        else:
            self.expected_checkpoint_folder = f"./models/{self.initial_model_name}"
            self.expected_model_folder = f"{self.expected_checkpoint_folder}/model/"
            self.expected_tokenizer_folder = f"{self.expected_checkpoint_folder}/tokenizer/"
            self.model_folderpath = self.expected_model_folder
            self.tokenizer_folderpath = self.expected_tokenizer_folder

        print(self.config)
        self.tokenizer = self.load_tokenizer(folderpath=self.tokenizer_folderpath)
        self.model = self.load_model(folderpath=self.model_folderpath)
        if is_to_apply_lora:
            self.model = self.get_lora_model(self.model)

        print(f'>>> tokenizing dataset...')
        self.tokenized_dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=list(self.dataset['train'][0])
        )
        print(f'>>> tokenized dataset: {self.tokenized_dataset["train"][0]}')

        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.config.MLM_PROBABILITY,
            return_tensors='pt',
        )
        exit()

    def save_data_splits(self):
        train_fp = f"./train_{self.dataset_name}"
        test_fp = f"./test_{self.dataset_name}"
        with open(train_fp, "w") as outf:
            for e in self.dataset['train']:
                outf.write(e['text'] + '\n')
        with open(test_fp, "w") as outf:
            for e in self.dataset['test']:
                outf.write(e['text'] + '\n')

    def dataset_name(self):
        if '/' in self.config.DATASET_NAME:
            dataset_name = self.config.DATASET_NAME.rstrip('/').split('/')[-1]
        else:
            dataset_name = self.config.DATASET_NAME
        return dataset_name

    def get_initial_model_name_from_checkpoint(self):
        if '/' in self.config.MODEL_CHECKPOINT:
            initial_model_name = self.config.MODEL_CHECKPOINT.rstrip('/').split('/')[-1]
        else:
            initial_model_name = self.config.MODEL_CHECKPOINT
        return initial_model_name

    def load_tokenizer(self, folderpath):
        print(folderpath)
        tokenizer = AutoTokenizer.from_pretrained(
            folderpath, local_files_only=True
        )
        return tokenizer

    def load_model(self, folderpath):
        model = AutoModelForMaskedLM.from_pretrained(
            folderpath, local_files_only=True
        )
        return model

    def load_dataset(self, dataset_name):
        expected_local_datasets_names_text_column = {
                "efcamdat": "text", 
                "EFCAMDAT": "text", 
                "celva"   : "text", 
                "CELVA"   : "text" 
        }
        # if dataset_name is a filepath that its a single training file
        # else we assume its a folder path
        if os.path.isfile(dataset_name):
            with open(dataset_name, encoding='utf-8') as inpf:
                texts = [sent.replace('\n', '') for sent in open(dataset_name, encoding='utf-8')]
                my_dict = {'text': texts}
                dataset = HF_Dataset.from_dict(my_dict)
                dataset = dataset.train_test_split(
                    test_size=0.1, shuffle=True, seed=200
                )
        elif expected_local_datasets_names_text_column\
                .get(dataset_name,False):
            # here we assume the expected dataset is in a folder structre.
            # those are datasets I expect to follow the typical folder structure i used for my projects
            # ./{datasets_folder}/{dataset}/tokenization_batch/{split}/*.json.compact.gz
            # if splti else ./{datasets_folder}/{dataset}/tokenization_batch/*.json.compact.gz 
            # each file is a json.compact.gz file of a batch of the total dataset
            # each can be loaded using srsly and should have the following fields (they can be nested objects:
            # text_id, text, text_metadata, sentences, tokens
            # if training_strategy is "FULL+LLM-TOKENIZE"
            # then load each batch json and get the "text field
            # if training_strategy is "RESUME+LLM-TOKENIZE"
            # then find the latest processed batch and load onwards getting "text" field
            # if training_strategy is "FULL+HUMAN-TOKENIZE" 
            # then load each batch json and get "tokens" which is an array of token objects
            print(f"{self.config.TRAINING_STRATEGY} == {TrainingStrategy.FULL_LLM_TOKENIZE}")
            if self.config.TRAINING_STRATEGY.value == TrainingStrategy.FULL_LLM_TOKENIZE.value:
                self.expected_folderpath = pathlib.Path("./") /\
                        f"{self.config.DATASET_FOLDER}"/\
                        f"{self.config.DATASET_NAME.upper()}"/\
                        f"tokenization_batch"/\
                        f"{self.config.SPLIT if self.config.SPLIT else ''}"
                expected_text_column = expected_local_datasets_names_text_column\
                                                    .get(dataset_name,False)
                # assuming is from folderpath, it's 
                # if htere is a SPLIT use only this split
                # if there is no split try getting /train /test
                # if splti else ./{datasets_folder}/{dataset}/tokenization_batch/*.json.compact.gz 
                # each file is a json.compact.gz file of a batch of the total dataset
                dataset = defaultdict(list)
                for f in self.expected_folderpath.iterdir(): 
                    with open(f) as inpf:
                        try:
                            data_dict = srsly.read_gzip_json(f)
                        except:
                            print(f)
                        dataset["text"].extend([instance_d["text"] for instance_d in data_dict.values()])
                dataset = HF_Dataset.from_dict(my_dict)
                dataset = dataset.train_test_split(
                    test_size=0.1, shuffle=True, seed=200
                )
                print("then load each batch json and get the text field")
            else:
                raise Error("TRAINING_STRATEGY seems to not be a valid one")
        else:
            dataset = hf_load_dataset(dataset_name)
        sample = dataset['train'].shuffle(seed=42).select(range(3))
        for row in sample:
            print(f"\n'>>> {row['text']}'")
        return dataset

    def tokenize_function(self, example):
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

    def get_lora_model(self, model):
        modules = self.find_all_linear_names(model)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=modules,
            bias="all",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        return model

    def find_all_linear_names(self, model):
        cls = torch.nn.modules.linear.Linear
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        print(f"LoRA module names: {list(lora_module_names)}")
        return list(lora_module_names)

    def train(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        print(torch.cuda.current_device())
        time.sleep(5)
        print(f'>>> Training model...')

        logging_steps = len(self.tokenized_dataset['train']) // self.config.BATCH_SIZE
        model_name = self.config.MODEL_CHECKPOINT.split('/')[-1]
        dataset_name = self.dataset_name
        finetuned_model_output_dir = f'./models/{model_name}-finetuned-{dataset_name}'

        training_args = HF_TrainingArguments(
            output_dir=finetuned_model_output_dir,
            num_train_epochs=10,
            overwrite_output_dir=True,
            logging_strategy='epoch',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=5e-5,
            weight_decay=0.01,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            push_to_hub=False,
            fp16=False,
            resume_from_checkpoint=True,
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
    
    # Load the config file and convert it into a dataclass instance
    with open(config_filepath) as inpf:
        config_dict = json.load(inpf)
        config = TrainerConfig(**{k.upper(): v for k, v in config_dict.items()})

    trainer = Trainer(config)
    trainer.train()
