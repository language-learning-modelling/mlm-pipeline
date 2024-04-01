import json
def load_config(config_filepath):
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
