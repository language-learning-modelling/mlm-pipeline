class CustomTrainer(Trainer):
    # def __init__(self, *args, **kwargs):
    #     print_steps = kwargs.pop('print_steps', 100)  # Set default to 100 steps
    #     super().__init__(*args, **kwargs)
    #     self.add_callback(PrintTrainingDataCallback(print_steps))

    def train(self, resume_from_checkpoint=None, **kwargs):
        if resume_from_checkpoint is not None:
            print(f"Loading checkpoint from {resume_from_checkpoint}")
            # Load checkpoint logic here
            if os.path.isdir(resume_from_checkpoint):
                trainer_state_fp = os.path.join(resume_from_checkpoint, "trainer_state.json")
                with open(trainer_state_fp) as inpf:
                    state_dict = json.load(inpf)
            print(state_dict)
            print(f"Resuming at global_step {self.state.global_step}")
        super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

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
