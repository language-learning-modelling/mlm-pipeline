import json
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

class Predictor(object):
    def __init__(self, config_filepath):
        self.config = self.load_config(config_filepath)

    def load_config(self, config_filepath):
        with open(config_filepath) as inpf:
            config = json.load(inpf)
            config = {k.upper(): v for k, v in config.items()}
            return config
            # self.__dict__.update(**config)

    def predict(self):
        model = AutoModelForMaskedLM.from_pretrained(
            self.config['MODEL_CHECKPOINT']
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['MODEL_CHECKPOINT']
        )
        model_num_parameters = model.num_parameters() / 1_000_000
        print(
            f"'>>> model number of parameters: {round(model_num_parameters)}M'"
        )

        inputs = tokenizer(self.config['TEXT'], return_tensors='pt')
        token_logits = model(**inputs).logits
        # Find the location of [MASK] and extract its logits
        mask_token_index = torch.where(
            inputs['input_ids'] == tokenizer.mask_token_id
        )[1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        # Pick the [MASK] candidates with the highest logits
        top_5_tokens = (
            torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        )
        print(f"{self.config['TEXT']}")
        for token in top_5_tokens:
            print(
                f"'>>> {self.config['TEXT'].replace(tokenizer.mask_token, tokenizer.decode([token]))}'"
            )


if __name__ == '__main__':
    import sys
    config_filepath = sys.argv[1]
    print(config_filepath)
    predictor = Predictor(config_filepath)
    predictor.predict()
