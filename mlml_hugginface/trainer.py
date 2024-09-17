import os
import json
from transformers import (
    Trainer,
    TrainerCallback,
)
import torch

def print_start_end(func):
    def wrapper(self, *args, **kwargs):
        # Dynamically determine the method name
        method_name = func.__name__

        print(f"Starting {method_name}")

        # Dynamically call the super method if it's not a static method or class method
        if hasattr(super(type(self), self), method_name):
            getattr(super(type(self), self), method_name)(*args, **kwargs)

        # Call the actual method
        result = func(self, *args, **kwargs)

        print(f"Ending {method_name}")
        input()
        return result
    return wrapper


