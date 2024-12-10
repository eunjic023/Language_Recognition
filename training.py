
from trainer import CustomTrainer
from metrics import cal_metrics
from loading import model, tokenized_train,tokenized_val
from transformers import TrainingArguments
from logg import WandbLoggingCallback,  SaveModelCallback
from datetime import datetime
import time
import wandb
import torch
import optuna

torch.cuda.empty_cache()

timestamp = datetime.now().strftime("%Y-%m%d-%H%M")
wandb.init(project="Language_Recognition", name="distilled_roberta_finetuning")
train_dataset = tokenized_train

def objective(trial):
    # 하이퍼파라미터 선택
    lr = trial.suggest_float("learning_rate", 0.00001, 5e-5, log=True)
    train_batch_size_range = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])

    for param in model.bert.encoder.layer[:4].parameters():
        param.requires_grad = False  

    training_args = TrainingArguments(
        output_dir=f"./results/{timestamp}",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=lr,
        per_device_train_batch_size=train_batch_size_range,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        report_to=["wandb"],
        gradient_accumulation_steps=4,
        bf16=True,
        load_best_model_at_end=True
        )

    trainer = CustomTrainer(model=model,
        args=training_args,
        compute_metrics=cal_metrics,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        callbacks=[WandbLoggingCallback(), SaveModelCallback()]
        )
    start_time = time.time()
    train_output = trainer.train()
    train_metrics = trainer.evaluate(eval_dataset=tokenized_train)
    val_metrics = trainer.evaluate(eval_dataset=tokenized_val)
    end_time=time.time()

    print("excuted time : ", end_time-start_time)
    wandb.log({
        "train_loss": train_metrics["eval_loss"],
        "train_accuracy": train_metrics["eval_accuracy"],
        "val_loss": val_metrics["eval_loss"],
        "val_accuracy": val_metrics["eval_accuracy"]
        })
    return val_metrics["eval_loss"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best hyperparameters:", study.best_params)
wandb.finish()
