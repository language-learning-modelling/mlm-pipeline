import os
import json
from transformers import (
    Trainer,
    TrainerCallback,
)
class CustomTrainer(Trainer):
    # def __init__(self, *args, **kwargs):
    #     print_steps = kwargs.pop('print_steps', 100)  # Set default to 100 steps
    #     super().__init__(*args, **kwargs)
    #     self.add_callback(PrintTrainingDataCallback(print_steps))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Training loop
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
        super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    # On the start of the training loop
    def on_train_begin(self):
        print("Custom training begin.")
        super().on_train_begin()

    # On the start of each epoch
    def on_epoch_begin(self):
        print("Custom epoch begin.")
        super().on_epoch_begin()

    # Training step
    def training_step(self, model, inputs):
        print("Custom training step started.")
        return super().training_step(model, inputs)

    # End of training step
    def training_step_end(self, loss):
        print("Custom training step ended.")
        return super().training_step_end(loss)

    # On the end of each epoch
    def on_epoch_end(self):
        print("Custom epoch end.")
        super().on_epoch_end()

    # On the end of the training loop
    def on_train_end(self):
        print("Custom training end.")
        super().on_train_end()

    # Evaluation loop start
    def on_evaluate(self):
        print("Custom evaluation begin.")
        super().on_evaluate()

    # Evaluation step
    def evaluation_step(self, model, inputs):
        print("Custom evaluation step started.")
        return super().evaluation_step(model, inputs)

    # End of evaluation step
    def evaluation_step_end(self, outputs):
        print("Custom evaluation step ended.")
        return super().evaluation_step_end(outputs)

    # Evaluation loop end
    def on_evaluate_end(self):
        print("Custom evaluation end.")
        super().on_evaluate_end()

    # Data collation (custom logic for batch processing)
    def data_collator(self, features):
        print("Custom data collator called.")
        return super().data_collator(features)

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
