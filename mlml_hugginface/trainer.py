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
        print(args)
        print(kwargs)
        print(self)


        # Call the actual method
        result = func(self, *args, **kwargs)

        # Dynamically call the super method if it's not a static method or class method
        if hasattr(super(type(self), self), method_name):
            getattr(super(type(self), self), method_name)(*args, **kwargs)

        print(f"Ending {method_name}")
        return result
    return wrapper


class CustomTrainer(Trainer):
    # def __init__(self, *args, **kwargs):
    #     print_steps = kwargs.pop('print_steps', 100)  # Set default to 100 steps
    #     super().__init__(*args, **kwargs)
    #     self.add_callback(PrintTrainingDataCallback(print_steps))

    @print_start_end
    def __init__(self, *args, **kwargs):
        print(kwargs)

    @print_start_end
    def train(self, resume_from_checkpoint=None, **kwargs):
        if resume_from_checkpoint is not None:
            print(f"Loading checkpoint from {resume_from_checkpoint}")
            # Load checkpoint logic here
            if os.path.isdir(resume_from_checkpoint):
                trainer_state_fp = os.path.join(resume_from_checkpoint, "trainer_state.json")
                with open(trainer_state_fp) as inpf:
                    state_dict = json.load(inpf)
            self.state.global_step = state_dict["global_step"] 
            print(state_dict)
            print(f"Resuming at global_step {self.state.global_step}")
        input("i was going to run the super()")
        # super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    # On the start of the training loop
    @print_start_end
    def on_train_begin(self):
        pass

    # On the start of each epoch
    @print_start_end
    def on_epoch_begin(self):
        pass

    @print_start_end
    def training_step(self, model, inputs):
        pass

    @print_start_end
    def training_step_end(self, loss):
        pass

    # On the end of each epoch
    @print_start_end
    def on_epoch_end(self):
        pass

    # On the end of the training loop
    @print_start_end
    def on_train_end(self):
        pass

    # Evaluation loop start
    @print_start_end
    def on_evaluate(self):
        pass

    # Evaluation step
    @print_start_end
    def evaluation_step(self, model, inputs):
        pass

    # End of evaluation step
    @print_start_end
    def evaluation_step_end(self, outputs):
        pass

    # Evaluation loop end
    @print_start_end
    def on_evaluate_end(self):
        pass

    # Data collation (custom logic for batch processing)
    @print_start_end
    def data_collator(self, features):
        pass

class PrintTrainingDataCallback(TrainerCallback):
    def __init__(self, print_steps):
        self.print_steps = print_steps

    def on_step_end(self, args, state, control, **kwargs):
        print("*"*100,"Training step ending","*"*100)
        step = state.global_step
        print(kwargs.keys())
        print(state)
        input()
        dataloader = kwargs.get("train_dataloader")
        for batch in dataloader:
            print(f"Step {step}:")
            print(batch)  # or print a sample of the batch data
            input()
            break
        print(step)
        input()

class SaveAtEndOfEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True  # Force save at the end of the epoch
