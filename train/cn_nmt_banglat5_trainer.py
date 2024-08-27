# -*- coding: utf-8 -*-
##########################################
"""# Config"""
##########################################

class CONFIG:
    # Model settings
    root_model_name = 'csebuetnlp/banglat5'
    # for reproducibility
    seed = 1234 # Change before each experiment
    # root_model_name = 'google/mt5-small'
    step_id = 1 # Change before each experiment count is from 1
    is_forward = False # Change before each experiment
    # Please set the output-directory according to your need
    if is_forward:
      output_dir = './BanglaT5-results/'
    else:
      output_dir = './BanglaT5-results/'
    device = "cuda"
    # device = "cpu" if you don't have any GPU

    # language # actual source = 'ck', target = 'bn'
    source_lang = 'bn'
    target_lang = 'ccp'

    # data params
    dataset_name = 'amlan107/chakma-nmt-complete-dataset' # test set + monlingual + parallel
    dev_val = 'amlan107/chakma-nmt-base-parallel-dev-set' # validation
    base_syn_name = 'amlan107/chakma-nmt-base-parallel-train-set' #12k base parallel data for training

    synthetic_dataset_name = 'amlan107/chakma-nmt-base-parallel-train-set'  #Initially for the first step train with the base parallel training data, in the second step train with the "synthetic_dataset_next"
    # synthetic data is generated from the monolingual dataset.

    if is_forward:
        synthetic_dataset_next = 'syn_true' + str(step_id) + '_bt5_seed' + str(seed) # base_parallel + synthetic produced from monolingual
    else:
        synthetic_dataset_next = 'syn_false' + str(step_id) + '_bt5_seed' + str(seed) # base_parallel + synthetic produced from monolingual
        # naming convention of sythetic_dataset_next --> syn_false/true (false->backward direction, true->forward direction)
    
    
    total_syn_to_generate = 50000 # Total number of samples we want to generate for synthetic data

    # Training settings
    max_length = 128
    train_batch_size = 16
    val_batch_size = 16
    lr = 0.0005
    lr_scheduler_type = "linear"
    warmup_ratio = 0.1
    train_epochs = 5 #5 # 15
    eval_strategy = "epoch"
    eval_steps = None
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
    num_workers = 8

    #prefix for T5

    use_prefix = False
    bn2ck = "Translate from Bangla to Chakma: "
    ck2bn = "Translate from Chakma to Bangla: "











import os
import sys

if not os.path.isdir(CONFIG.output_dir):
  os.mkdir(CONFIG.output_dir)

CONFIG.output_dir = CONFIG.output_dir + 'seed-' + str(CONFIG.seed)
if not os.path.isdir(CONFIG.output_dir):
  os.mkdir(CONFIG.output_dir)

if CONFIG.is_forward:
  CONFIG.output_dir = CONFIG.output_dir + '/step-' + str(CONFIG.step_id) + '_Forward-B2C'
  if not os.path.isdir(CONFIG.output_dir):
    os.mkdir(CONFIG.output_dir)
  CONFIG.output_dir = CONFIG.output_dir + '/Banglat5-v1'
  if not os.path.isdir(CONFIG.output_dir):
    os.mkdir(CONFIG.output_dir)
else:
  CONFIG.output_dir = CONFIG.output_dir + '/step-' + str(CONFIG.step_id) + '_Backward-C2B'
  if not os.path.isdir(CONFIG.output_dir):
    os.mkdir(CONFIG.output_dir)











#############################################
"""# Imports"""
#############################################

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import random
import re
import string

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from datasets import load_metric, config
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
from pathlib import Path






#########################################################################
"""# Linking Important Directories and Creating Dataset path"""
#########################################################################

sys.path.append(os.getcwd() + '/../utils')
from ck2bn_bn2ck_phonetic import ck2bn_list, bn2ck_list, ck2bn, bn2ck

dataset_path = os.getcwd() + '/Dataset'
if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
config.DOWNLOADED_DATASETS_PATH = Path(dataset_path)







############################################################
"""# Deterministic settings"""
############################################################

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







##################################################
"""# Suppress Warnings"""
##################################################

# If you want to suppress all warnings
import warnings
warnings.filterwarnings("ignore")








############################################################
"""# Dataset"""
############################################################

print(f'Downloading Dataset: {CONFIG.dataset_name}\n')
RAW_DATASET = load_dataset(CONFIG.dataset_name)

def process_function(features):

    # buetcsenlp normalization
    inputs = [normalize(sen) if sen != None else None for sen in features[CONFIG.source_lang]]
    targets = [normalize(sen) if sen != None else None for sen in features[CONFIG.target_lang]]

    result = dict()
    result[CONFIG.source_lang] = inputs
    result[CONFIG.target_lang] = targets

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





print(f'Downloading Dataset: {CONFIG.base_syn_name}\n')
BASE_SYN = load_dataset(CONFIG.base_syn_name)

def process_function(features):

    # buetcsenlp normalization
    inputs = [normalize(sen) if sen != None else None for sen in features[CONFIG.source_lang]]
    targets = [normalize(sen) if sen != None else None for sen in features[CONFIG.target_lang]]

    result = dict()
    result[CONFIG.source_lang] = inputs
    result[CONFIG.target_lang] = targets

    return result

BASE_SYN['train'] = BASE_SYN['train'].map(
    process_function,
    batched=True
)

print(BASE_SYN)







print(f'Downloading Dataset: {CONFIG.synthetic_dataset_name}\n')
SYN_DATASET = load_dataset(CONFIG.synthetic_dataset_name)
print(SYN_DATASET)

def process_function(features):

    # buetcsenlp normalization
    inputs = [normalize(sen) if sen != None else None for sen in features[CONFIG.source_lang]]
    targets = [normalize(sen) if sen != None else None for sen in features[CONFIG.target_lang]]

    result = dict()
    result[CONFIG.source_lang] = inputs
    result[CONFIG.target_lang] = targets

    return result

SYN_DATASET['train'] = SYN_DATASET['train'].map(
    process_function,
    batched=True
)






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

DEV_VALIDATION_DATASET['dev_val'] = DEV_VALIDATION_DATASET['dev_val'].map(
    process_function,
    batched=True
)

print(DEV_VALIDATION_DATASET)







################################################
"""# Tokenizer"""
################################################

TOKENIZER = AutoTokenizer.from_pretrained(CONFIG.root_model_name, use_fast=False)









##################################################
"""# Tokenization"""
##################################################

def tokenization(features, tokenizer = None, forward_train = True):
    if forward_train:
        prefix = ''
        if CONFIG.use_prefix:
            prefix = CONFIG.bn2ck

        s_ = [prefix + text for text in features[CONFIG.source_lang]]
        t_ = ck2bn_list(features[CONFIG.target_lang])

    else:
        prefix = ''
        if CONFIG.use_prefix:
            prefix = CONFIG.ck2bn

        s_ = [prefix + text for text in ck2bn_list(features[CONFIG.target_lang])]
        t_ = features[CONFIG.source_lang]

    source = tokenizer(s_, max_length=CONFIG.max_length, truncation=True)
    targets = tokenizer(t_, max_length=CONFIG.max_length, truncation=True)

    source['labels'] = targets.input_ids
    return source

tokenized_train = SYN_DATASET['train'].map(partial(tokenization, tokenizer = TOKENIZER, forward_train = CONFIG.is_forward),
                                  batched=True,
                                  remove_columns=SYN_DATASET['train'].column_names)

tokenized_val = DEV_VALIDATION_DATASET['dev_val'].map(partial(tokenization, tokenizer = TOKENIZER, forward_train = CONFIG.is_forward),
                                 batched=True,
                                 remove_columns=DEV_VALIDATION_DATASET['dev_val'].column_names)

tokenized_test = BENCHMARK_DATASET.map(partial(tokenization, tokenizer = TOKENIZER, forward_train = CONFIG.is_forward),
                                 batched=True,
                                 remove_columns=BENCHMARK_DATASET.column_names)

tokenized_train.set_format("torch")
tokenized_val.set_format("torch")
tokenized_test.set_format("torch")

def mono_tokenization(features, tokenizer = None, forward_train = True):
    if forward_train:
        prefix = ''
        if CONFIG.use_prefix:
            prefix = CONFIG.bn2ck

        s_ = [prefix + text for text in features[CONFIG.source_lang]]

    else:
        prefix = ''
        if CONFIG.use_prefix:
            prefix = CONFIG.ck2bn

        s_ = [prefix + text for text in ck2bn_list(features[CONFIG.target_lang])]

    source = tokenizer(s_, max_length=CONFIG.max_length, truncation=True)
    return source

temp_lang = CONFIG.source_lang if CONFIG.is_forward else CONFIG.target_lang

temp_sentences = []
syn_cnt = 0
for sentence in RAW_DATASET['monolingual'][temp_lang]:
        if sentence == None or syn_cnt == CONFIG.total_syn_to_generate:
            break
        temp_sentences.append(sentence)
        syn_cnt+=1

temp_dataset =  Dataset.from_dict({temp_lang : temp_sentences})


tokenized_mono =temp_dataset.map(partial(mono_tokenization, tokenizer = TOKENIZER, forward_train = CONFIG.is_forward),
                                 batched=True,
                                 remove_columns=temp_dataset.column_names)

tokenized_mono.set_format("torch")
print(tokenized_mono)










##################################################
"""# Model"""
##################################################

model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG.root_model_name)
## if you want to train a pre-trained model that is placed locally
# model = AutoModelForSeq2SeqLM.from_pretrained('./BanglaT5-results/seed-0/step-2_Forward-B2C/Banglat5-v1/checkpoint-17125', local_files_only=True)
model = model.to(CONFIG.device)
# Fixing Contigious tensor error
for param in model.parameters():
    param.data = param.data.contiguous()
print(model)









###############################################
"""# Data Collator"""
###############################################

data_collator = DataCollatorForSeq2Seq(TOKENIZER, model=model)










##############################################
"""# Metrics and function"""
##############################################

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












##############################################
"""# Configs & Trainer"""
##############################################
####################### Trainer

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
    warmup_ratio=CONFIG.warmup_ratio,
    per_device_train_batch_size=CONFIG.train_batch_size,
    per_device_eval_batch_size=CONFIG.val_batch_size,
    num_train_epochs=CONFIG.train_epochs,
    weight_decay=CONFIG.weight_decay,
    save_strategy=CONFIG.eval_strategy,
    evaluation_strategy=CONFIG.eval_strategy,
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

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset= tokenized_train, # tokenized_datasets['train'], # tokenized_train,
    eval_dataset= {'val' : tokenized_val, 'test': tokenized_test},# tokenized_datasets['validation'], # tokenized_val,
    tokenizer=TOKENIZER,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)







##########################################
"""# Main Train"""
##########################################
trainer.train()











###########################################################################
"""# To Generate Synthetic Data and Upload to HuggingFace """
###########################################################################

# predictions = trainer.predict(tokenized_mono, max_length = CONFIG.max_length, num_beams = CONFIG.num_beans)

# # predictions.predictions.shape

# pred_decoded = TOKENIZER.batch_decode(predictions.predictions, skip_special_tokens=True)

# def upload_synthetic_data(syn_new_data):
#     ## create data
#     parallelDataset = Dataset.from_dict({
#                                         CONFIG.source_lang: BASE_SYN['train'][CONFIG.source_lang] + syn_new_data[CONFIG.source_lang],
#                                         CONFIG.target_lang:  BASE_SYN['train'][CONFIG.target_lang] + syn_new_data[CONFIG.target_lang]
#                                         })

#     ## create dataset
#     combined_new_syn_dataset = DatasetDict({'train':  parallelDataset})

#     ## upload
#     combined_new_syn_dataset.push_to_hub(CONFIG.synthetic_dataset_next)

# # Huggingface Login:

# notebook_login()

# if CONFIG.is_forward:
#     upload_synthetic_data({CONFIG.source_lang: temp_sentences, CONFIG.target_lang: pred_decoded})
# else:
#     upload_synthetic_data({CONFIG.source_lang: pred_decoded, CONFIG.target_lang: temp_sentences})











######################################################################################
"""# TO TEST A SINGLE SENTENCE WITH THE TRAINED MODEL """
######################################################################################

# model = AutoModelForSeq2SeqLM.from_pretrained('./BanglaT5-results/seed-1234/step-4_Forward-B2C/Banglat5-v1/checkpoint-17125')

# model = model.to('cuda')

# RAW_DATASET['benchmark']['en'][596]

# # Bangla to chakma
# # এই কলমটির দাম কত?
# # আমি সকাল ৮ টায় কাজে যাব।
# # পরিক্ষা ভাল হয়েছে।


# # Chakma to Bangla
# # 𑄖𑄪𑄃𑄨 𑄃𑄨𑄇𑄴𑄇𑄬 𑄦𑄪𑄘𑄪 𑄃𑄉𑄧𑄌𑄴𑅃 ?
# # 𑄟𑄪𑄃𑄨 𑄞𑄖𑄴 𑄦𑄋𑄴𑄢𑄴 𑅁
# # 𑄃𑄬𑄃𑄨 𑄦𑄟𑄴 𑄃𑄚𑄴 𑄉𑄧𑄢𑄨 𑄗𑄮𑄎𑄴 𑅁


# # English to Chakma Zeroshot
# text = 'I was cleaning the house all evening after I saw how dirty it was.'
# text = ck2bn(text)
# print(text)
# input_ids = TOKENIZER(normalize(text), return_tensors="pt", truncation = True, max_length=128).input_ids.to("cuda")

# generated_tokens = model.generate(input_ids, num_beams=5, max_length=128, num_return_sequences=1)

# decoded_tokens = TOKENIZER.batch_decode(generated_tokens, skip_special_tokens=False)[0]

# print(decoded_tokens)

# print(bn2ck(decoded_tokens))