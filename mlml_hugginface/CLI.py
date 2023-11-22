from typing import Optional

import typer
import sys

app = typer.Typer()


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


if __name__ == "__main__":
    app()
