from typing import Optional

import typer
import sys

app = typer.Typer()


@app.command()
def generate(cfgf: str):
    from change import NewModel
    from transformers import BertConfig, AutoTokenizer

    config = BertConfig(model_name='bert-base-uncased')
    model = NewModel(config)
    model.save_pretrained("./models/new-model-27")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.save_pretrained("./models/new-model-27")


@app.command()
def train(cfgf: str):
    from train import Trainer

    trainer = Trainer(cfgf)
    trainer.train()


@app.command()
def predict(cfgf: str):
    from predict import Predictor

    print(cfgf)
    predictor = Predictor(cfgf)
    predictor.predict()


if __name__ == '__main__':
    app()
