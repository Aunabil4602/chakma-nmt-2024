# -*- coding: utf-8 -*-
# ===========================================
""" CONFIGURATION """
# ===========================================

class CONFIG:
    # Model settings
    root_model_name = 'csebuetnlp/banglat5'
    # root_model_name = 'google/mt5-small'
    output_dir = 'multi-banglat5-v3'
    device = "cuda"

    # language # actual source = 'ck', target = 'bn'
    source_lang = 'bn' # no need
    target_lang = 'ck' # no need

    # data params
    dataset_name = 'amlan107/xyz'
    dev_val = 'amlan107/multilingual_dv'
    train_set = 'amlan107/ccp_nmt_multilingual_train_only'

    # Training settings
    max_length = 128
    train_batch_size = 16
    val_batch_size = 16
    lr = 0.0005
    lr_scheduler_type = "linear"
    warmup_steps = 1000
    train_steps = 7000
    eval_strategy = "steps"
    eval_steps = 1000
    save_strategy = "steps"
    save_steps = 1000
    weight_decay = 0.01
    label_smoothing_factor = 0.3
    save_total_limit = 10
    metric_for_best_model = "bleu"
    fp16=False

    # beam search
    num_beans = 5
    top_k = 10
    num_return_sequences = 1
    predict_with_generate = True

    # Other settings
    seed = 42
    num_workers = 8

    #prefix for T5
    use_prefix = False
    any2ck = "<ccp> "
    any2bn = "<bn> "
    any2en = "<en> "




################################################
"""# Imports"""
################################################

import os
import sys
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import random
import re
import string

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from datasets import load_metric
from normalizer import normalize
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments, Seq2SeqTrainingArguments

#####

from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from normalizer import normalize
from transformers import (
    AdamW,
    DataCollatorForSeq2Seq,
    MarianTokenizer,
    MarianMTModel,
    MarianConfig,
    get_scheduler,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    Seq2SeqTrainer
)

# import os
import sentencepiece as spm
import random
import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import operator
from queue import PriorityQueue
import json
from functools import partial
from huggingface_hub import notebook_login






#######################################################################
"""# Linking and Creating Important Directories"""
#######################################################################

sys.path.append(os.getcwd() + '/../utils')
from ck2bn_bn2ck_phonetic import ck2bn_list, bn2ck_list, ck2bn, bn2ck

if not os.path.isdir(CONFIG.output_dir):
    os.mkdir(CONFIG.output_dir)










####################################################
"""# Deterministic settings"""
####################################################

def set_seed(seed=42, loader=None):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass

set_seed(seed=CONFIG.seed)











#########################################
"""# Dataset"""
#########################################

print(f'Downloading Dataset: {CONFIG.dataset_name}\n')
RAW_DATASET = load_dataset(CONFIG.dataset_name)

def process_function(features):


    result = dict()
    result['ck'] = [normalize(sen) if sen != None else None for sen in features['ck']]
    result['en'] = [normalize(sen) if sen != None else None for sen in features['en']]
    result['bn'] = [normalize(sen) if sen != None else None for sen in features['bn']]

    return result

RAW_DATASET['parallel'] = RAW_DATASET['parallel'].map(
    process_function,
    batched=True
)

RAW_DATASET['monolingual'] = RAW_DATASET['monolingual'].map(
    process_function,
    batched=True
)

RAW_DATASET['benchmark'] = RAW_DATASET['benchmark'].map(
    process_function,
    batched=True
)

print(RAW_DATASET)
BENCHMARK_DATASET = RAW_DATASET['benchmark']







print(f'Downloading Dataset: {CONFIG.train_set}\n')
BASE_TRAIN = load_dataset(CONFIG.train_set)

def process_function(features):

    # buetcsenlp normalization
    inputs = [normalize(sen) if sen != None else None for sen in features['src']]
    targets = [normalize(sen) if sen != None else None for sen in features['tgt']]

    result = dict()
    result['src'] = inputs
    result['tgt'] = targets

    return result

BASE_TRAIN['train'] = BASE_TRAIN['train'].map(
    process_function,
    batched=True
)

print(BASE_TRAIN)








print(f'Downloading Dataset: {CONFIG.dev_val}\n')
DEV_VALIDATION_DATASET = load_dataset(CONFIG.dev_val)
def process_function(features):

    # buetcsenlp normalization
    inputs = [normalize(sen) if sen != None else None for sen in features[CONFIG.source_lang]]
    targets = [normalize(sen) if sen != None else None for sen in features[CONFIG.target_lang]]

    result = dict()
    result[CONFIG.source_lang] = inputs
    result[CONFIG.target_lang] = targets

    return result

DEV_VALIDATION_DATASET['parallel'] = DEV_VALIDATION_DATASET['parallel'].map(
    process_function,
    batched=True
)
print(DEV_VALIDATION_DATASET)







##########################################
"""# Tokenizer"""
##########################################

TOKENIZER = AutoTokenizer.from_pretrained(CONFIG.root_model_name, use_fast=False)








###############################################
"""# Tokenization"""
###############################################

def tokenization_train(features, tokenizer = None):
    source = tokenizer(features['src'], max_length=CONFIG.max_length, truncation=True)
    targets = tokenizer(features['tgt'], max_length=CONFIG.max_length, truncation=True)

    source['labels'] = targets.input_ids
    return source

def tokenization_dev(features, tokenizer = None, forward_train = True):
    if forward_train:
        prefix = CONFIG.any2ck

        s_ = [prefix + text for text in features[CONFIG.source_lang]]
        t_ = ck2bn_list(features[CONFIG.target_lang])

    else:
        prefix = CONFIG.any2bn

        s_ = [prefix + text for text in ck2bn_list(features[CONFIG.target_lang])]
        t_ = features[CONFIG.source_lang]

    source = tokenizer(s_, max_length=CONFIG.max_length, truncation=True)
    targets = tokenizer(t_, max_length=CONFIG.max_length, truncation=True)

    source['labels'] = targets.input_ids
    return source

def tokenization_dev_english(features, tokenizer = None, forward_train = True):
    if forward_train:
        prefix = CONFIG.any2ck

        s_ = [prefix + text for text in features['en']]
        t_ = ck2bn_list(features['ck'])

    else:
        prefix = CONFIG.any2en

        s_ = [prefix + text for text in ck2bn_list(features['ck'])]
        t_ = features['en']

    source = tokenizer(s_, max_length=CONFIG.max_length, truncation=True)
    targets = tokenizer(t_, max_length=CONFIG.max_length, truncation=True)

    source['labels'] = targets.input_ids
    return source

tokenized_train = BASE_TRAIN['train'].map(partial(tokenization_train, tokenizer = TOKENIZER),
                                  batched=True,
                                  remove_columns=BASE_TRAIN['train'].column_names)

tokenized_val_forward = DEV_VALIDATION_DATASET['parallel'].map(partial(tokenization_dev, tokenizer = TOKENIZER, forward_train = True),
                                 batched=True,
                                 remove_columns=DEV_VALIDATION_DATASET['parallel'].column_names)

tokenized_val_backward = DEV_VALIDATION_DATASET['parallel'].map(partial(tokenization_dev, tokenizer = TOKENIZER, forward_train = False),
                                 batched=True,
                                 remove_columns=DEV_VALIDATION_DATASET['parallel'].column_names)

tokenized_test_forward = BENCHMARK_DATASET.map(partial(tokenization_dev, tokenizer = TOKENIZER, forward_train = True),
                                 batched=True,
                                 remove_columns=BENCHMARK_DATASET.column_names)

tokenized_test_backward = BENCHMARK_DATASET.map(partial(tokenization_dev, tokenizer = TOKENIZER, forward_train = False),
                                 batched=True,
                                 remove_columns=BENCHMARK_DATASET.column_names)

tokenized_test_en_forward = BENCHMARK_DATASET.map(partial(tokenization_dev_english, tokenizer = TOKENIZER, forward_train = True),
                                 batched=True,
                                 remove_columns=BENCHMARK_DATASET.column_names)

tokenized_test_en_backward = BENCHMARK_DATASET.map(partial(tokenization_dev_english, tokenizer = TOKENIZER, forward_train = False),
                                 batched=True,
                                 remove_columns=BENCHMARK_DATASET.column_names)

tokenized_train.set_format("torch")
tokenized_val_forward.set_format("torch")
tokenized_val_backward.set_format("torch")
tokenized_test_forward.set_format("torch")
tokenized_test_backward.set_format("torch")
tokenized_test_en_forward.set_format("torch")
tokenized_test_en_backward.set_format("torch")











#########################################
"""# Model"""
#########################################
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG.root_model_name)
model = model.to(CONFIG.device)
# Fixing Contigious tensor error
for param in model.parameters():
    param.data = param.data.contiguous()
print(model)











#######################################
"""# Data Collator"""
#######################################
data_collator = DataCollatorForSeq2Seq(TOKENIZER, model=model)














######################################
"""# Metrics and function"""
######################################

bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
meteor_metric = evaluate.load("meteor")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    # preds[preds > 32099] = TOKENIZER.pad_token_id
    decoded_preds = TOKENIZER.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

    # return {"bleu": bleu_result["score"]}
    return {"bleu": bleu_result["score"], "chrf": chrf_result["score"], "meteor": meteor_result["meteor"]}











######################################
"""# Configs & Trainer"""
######################################

generation_config = GenerationConfig(
    max_length = CONFIG.max_length,
    early_stopping = True,
    do_sample = True,
    top_k = CONFIG.top_k,
    # max_time = 10, # seconds
    num_beams = CONFIG.num_beans,
    num_return_sequences = CONFIG.num_return_sequences,
    pad_token_id = TOKENIZER.pad_token_id,
    bos_token_id = TOKENIZER.bos_token_id,
    eos_token_id = TOKENIZER.eos_token_id,
    decoder_start_token_id = TOKENIZER.pad_token_id
)

training_args = Seq2SeqTrainingArguments(
    # common args
    output_dir=CONFIG.output_dir,
    learning_rate=CONFIG.lr,
    lr_scheduler_type=CONFIG.lr_scheduler_type,
    warmup_steps=CONFIG.warmup_steps,
    per_device_train_batch_size=CONFIG.train_batch_size,
    per_device_eval_batch_size=CONFIG.val_batch_size,
    max_steps=CONFIG.train_steps,
    eval_strategy=CONFIG.eval_strategy,
    save_strategy=CONFIG.save_strategy,
    save_steps=CONFIG.save_steps,
    eval_steps=CONFIG.eval_steps,
    weight_decay=CONFIG.weight_decay,
    seed=CONFIG.seed,
    label_smoothing_factor = CONFIG.label_smoothing_factor,
    save_total_limit = CONFIG.save_total_limit,
    fp16=CONFIG.fp16,
    dataloader_num_workers=CONFIG.num_workers,
    load_best_model_at_end=False,
    # metric_for_best_model=CONFIG.metric_for_best_model,
    # greater_is_better=True,
    push_to_hub=False,

    # seq2seq specific
    predict_with_generate = CONFIG.predict_with_generate,
    generation_max_length = CONFIG.max_length,
    generation_num_beams = CONFIG.num_beans,
    generation_config = generation_config
)

evaluation_set = {
    'val': tokenized_val_forward,
    'val_backward': tokenized_val_backward,
    'test': tokenized_test_forward,
    'test_backward': tokenized_test_backward,
    'test_en': tokenized_test_en_forward,
    'test_en_backward': tokenized_test_en_backward
}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset= tokenized_train, # tokenized_datasets['train'], # tokenized_train,
    eval_dataset= evaluation_set,
    tokenizer=TOKENIZER,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)







############################################
"""# Main Train"""
############################################
trainer.train()







"""# Generate"""
# shutil.copytree('/content/multi-banglat5-v3', '/content/drive/MyDrive/cl-nmt-training-models/multi-banglat5-v3')