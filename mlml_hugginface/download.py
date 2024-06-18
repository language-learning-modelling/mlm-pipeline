import json
from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from .utils import  load_config

class DownloadConfig:
    def __init__(self, MODEL_CHECKPOINT):
        self.MODEL_CHECKPOINT = MODEL_CHECKPOINT

class Downloader:
    def __init__(self, cfg_fp):
        config_dict = load_config(cfg_fp)
        self.downloadConfig = DownloadConfig(**config_dict)
        print(f'>>> LOADED CONFIG')
        print(config_dict)
             
    def downloadLocally(self):
        base_outputDir = Path(f'./models/{self.downloadConfig.MODEL_CHECKPOINT}') 
        model_outputDir = Path(base_outputDir / 'model') 
        tokenizer_outputDir = Path(base_outputDir / 'tokenizer') 
        if base_outputDir.exists() and base_outputDir.is_dir():
            return 
        else:
            base_outputDir.mkdir(parents=True,exist_ok=True)
            model_outputDir.mkdir(parents=True,exist_ok=True) 
            tokenizer_outputDir.mkdir(parents=True,exist_ok=True) 

        model = AutoModelForMaskedLM.from_pretrained(
            self.downloadConfig.MODEL_CHECKPOINT
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.downloadConfig.MODEL_CHECKPOINT
        )
        model.save_pretrained(model_outputDir)
        tokenizer.save_pretrained(tokenizer_outputDir)
