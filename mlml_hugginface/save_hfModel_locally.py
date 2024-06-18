import os
import json
from transformers import AutoModelForMaskedLM, AutoTokenizer
from .utils import  load_config
import torch


class DownloadConfig:
    MODEL_CHECKPOINT = 'bert-base-uncased'

class Downloader:
    def __init__(self, cfg_fp):
             
    def downloadLocally(self):
        model = AutoModelForMaskedLM.from_pretrained(
            config['MODEL_CHECKPOINT']
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config['MODEL_CHECKPOINT']
        )
        os.system(f"mkdir -p ./{self.model_checkpoint}/model/")
        os.system(f"mkdir -p ./{self.model_checkpoint}/tokenizer/")
        model.save_pretained(f'./{self.model_checkpoint}/model/')
        tokenizer.save_pretained(f'./{self.model_checkpoint}/tokenizer/')
