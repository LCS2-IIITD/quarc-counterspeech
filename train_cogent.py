from modeling import COGENT, Seq2SeqTrainer
from utils import Config, extract_suffix, get_params, get_cogent_paths, save_cogent_components
from process_data_cogent import process_data, compute_metrics

import argparse
import wandb
import torch
from torch import nn
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer
)


def initialize(config_path):
    config_params = Config(config_fpath=config_path)
    suffix = extract_suffix(config_path)
    
    MODEL_PATH = config_params.cogent_model_init
    TOKENIZER_PATH = config_params.cogent_model_init
    
    BASE_PATH = 'models/'

    SAVE_PATH, LOGGING_PATH, SAVE_MODEL_PATH, wandb_runname = get_cogent_paths(BASE_PATH, suffix)
    
    batch_size = config_params.batch_size
    lr = config_params.lr
    
    wandb_run = wandb.init(
    project="quarc",
    config={
        "per_device_train_batch_size": batch_size,
        "learning_rate": lr})
    
    wandb_run.name = wandb_runname
    print(f']INFO] The W&B run name is: {wandb_runname}')
    
    return MODEL_PATH, TOKENIZER_PATH, SAVE_PATH, LOGGING_PATH, \
                SAVE_MODEL_PATH, config_params, batch_size, lr
    

def train_cogent(config_path):
    MODEL_PATH, TOKENIZER_PATH, SAVE_PATH, \
            LOGGING_PATH, SAVE_MODEL_PATH, config_params, \
            batch_size, lr = initialize(config_path)
            
    tokenized_datasets, tokenizer = process_data(config_params=config_params, 
                                        tokenizer_path=TOKENIZER_PATH)
    
    epochs = config_params.cogent_epochs
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'      
    model = COGENT.from_pretrained(MODEL_PATH).to(DEVICE)
    
    model.update(config_params=config_params)

    get_params(model=model)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    
    args = Seq2SeqTrainingArguments(
        output_dir=SAVE_PATH,
        learning_rate=lr,
        do_train = True,
        do_eval = True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.005,
        save_total_limit=3,
        load_best_model_at_end=True,
        num_train_epochs=epochs,
        predict_with_generate=True,
        # fp16=True,
        logging_dir=LOGGING_PATH,
        logging_steps=300,
        save_steps=600,
        metric_for_best_model='rougeLsum',
        greater_is_better=True,
        report_to = "wandb",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,)
    
    print(trainer.evaluate())
    print("\n"*2+"="*60+"\n"*2)
    print(trainer.train())
    print("\n"*2+"="*60+"\n"*2)
    print(trainer.evaluate())
    
    trainer.save_model(SAVE_MODEL_PATH)
    
    save_cogent_components(model=model, save_path=SAVE_MODEL_PATH)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default="configs/config_base.yaml", help="Path to the config file")

    args = parser.parse_args()
    train_cogent(config_path=args.config_path)