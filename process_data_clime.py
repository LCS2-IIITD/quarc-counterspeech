from utils.file_names import hs_col, cs_col, target_col, category_col
from utils import load_mappers

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import nltk
import numpy as np


tokenizer = None
target_mapper, category_mapper = load_mappers()
max_input_length = 256
max_target_length = 256
metric = load_metric("rouge")


def preprocess_function(examples):
    inputs = [doc for doc in examples[cs_col]]
    model_inputs = tokenizer(inputs, padding='max_length', max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[cs_col], padding='max_length', max_length=max_target_length, truncation=True)
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]

    model_inputs['targets'] = [target_mapper[target] for target in examples[target_col]]
    model_inputs['categories'] = [category_mapper[category] for category in examples[category_col]]

    inputs1 = [doc for doc in examples[hs_col]]
    model_inputs_1 = tokenizer(inputs1, padding='max_length', max_length=max_input_length, truncation=True)

    model_inputs['hs'] = model_inputs_1['input_ids']

    return model_inputs


def process_data(config_params, tokenizer_path):
    global tokenizer
    TRAIN_FILE_PATH = config_params.train_path
    VAL_FILE_PATH = config_params.val_path
    
    data_files = {}
    data_files["train"] = TRAIN_FILE_PATH
    data_files["validation"] = VAL_FILE_PATH
    
    raw_datasets = load_dataset("csv", data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)
    return tokenized_datasets, tokenizer


def compute_metrics(eval_pred):
    global tokenizer
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}
    