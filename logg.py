from transformers import  TrainingArguments,TrainerCallback
import wandb
import os

class WandbLoggingCallback(TrainerCallback):
    #from transformers import TrainerCallback
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs)


class SaveModelCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Save the model at the end of each epoch
        output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{state.epoch}")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        print(f"Model saved at {output_dir}")