# -*- coding: utf-8 -*-

## model's and source training code https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
## attention used is Luong attention https://arxiv.org/pdf/1508.04025.pdf
## Beam Search source code: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py

#############################################################
"""# CONFIGURATION """
#############################################################

class CONFIG:
    # version and outputs
    version = '-v228'
    root_output_dir = './cl-nmt-training-models/' # set the output directory
    is_forward = False # direction of the training. true means forward direction training: source to target training
    run_id = 0 # to track with the back-translation steps. starting-with 0
    model_type = 'rnn' # 'trans' or 'rnn'
    seed = 0 ### when the hyperparameters are complete. return the avg. bleu score of 5 different seeds.

    # data params
    dataset_name = 'amlan107/chakma-nmt-complete-dataset' # test set + monlingual + parallel
    dev_val = 'amlan107/chakma-nmt-base-parallel-dev-set' # validation
    base_syn_name = 'amlan107/chakma-nmt-base-parallel-train-set' #12k base parallel data for training

    synthetic_dataset_name = 'amlan107/chakma-nmt-base-parallel-train-set'  #Initially for the first step train with the base parallel training data, in the second step train with the "synthetic_dataset_next"
    
    if is_forward:
        synthetic_dataset_next = 'syn_true' + str(run_id) + '_' + model_type + '_seed' + str(seed) # base_parallel + synthetic produced from monolingual
    else:
        synthetic_dataset_next = 'syn_false' + str(run_id) + '_' + model_type + '_seed' + str(seed) # base_parallel + synthetic produced from monolingual
        # naming convention of sythetic_dataset_next --> syn_false/true (false->backward direction, true->forward direction)
    
    total_syn_to_generate = 50000

    # language
    source_lang = 'bn'
    target_lang = 'ccp'

    # model params rnn
    rnn_hidden_size = 1024
    rnn_embedding_size = 512
    rnn_enc_dec_num_layer = 1
    rnn_tie_decoder_embedding = True
    rnn_layer_normalization = True
    rnn_hidden_embedding_dropout = 0.3
    rnn_src_target_word_dropout = 0.3
    rnn_weight_init_std = 0.1

    # model params transformers
    trans_layers = 1
    trans_ff_rim = 512
    trans_att_heads = 1
    trans_dropout = 0.2
    trans_actv_dropout = 0.2
    trans_layer_dropout = 0.1
    trans_d_model = 512

    # tokenizer params
    vocab_size = 2000 # rnn 2000 # trans 10000
    unk_token_id = 0
    sos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3

    # training params
    optimizer = "adam"
    max_length = 128 # token length
    train_batch_size = 16 # rnn 16 # trans 32
    validation_batch_size = 32 # same for both trans and rnn
    train_test_split_ratio = 0.8 # not used
    learning_rate = 0.0005 # trans 0.0001 # rnn 0.0005
    num_train_epochs = 1000 # of no use now
    num_train_steps = 20000 # base train 20000 # with synthetic data 91000 # 4.55x
    num_warmup_steps =  4000 # base train 4000 # with synthetic data 18000 # 4.55x
    label_smoothing = 0.2 # trans 0.5 # rnn 0.2
    early_stoppings = 10
    keep_top_models = 10
    lr_gamma = 0.00009 # of no use now
    lr_update_every = 370 # of no use now
    clip_grad = 1.0

    # beam search
    use_custom_beam = True
    beam_width = 5
    top_k = 1
    beam_endpoints = 5
    queue_limit=5000

    # other
    on_the_fly_spm = False
    device = "cuda"
    # device = "cpu" if you don't have any GPU








##############################################################
"""# To Suppress All Warnings """
##############################################################

# Please comment out this part if you want to see the warnings
# If you want to suppress all warnings
import warnings
warnings.filterwarnings("ignore")





######################################################
"""# Linking the Necessary Directories """
######################################################

import os
import sys
sys.path.append(os.getcwd() + "/../utils")








#######################################################
"""# Imports"""
#######################################################

from datetime import datetime
from ck2bn_bn2ck_phonetic import *
from datasets import load_dataset, Dataset, DatasetDict, config
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
    get_scheduler
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






#####################################################################
"""# Training Version and DIrectory Setup """
#####################################################################

# set training destination folder
TODAY_DATE = datetime.today().strftime('%Y-%m-%d')

DESTINATION = CONFIG.root_output_dir + TODAY_DATE + CONFIG.version + '-rnn-step' + str(CONFIG.run_id) + '-forward-seed' + str(CONFIG.seed) + '/'

if not os.path.isdir(CONFIG.root_output_dir):
    os.mkdir(CONFIG.root_output_dir)
if not os.path.isdir(DESTINATION):
    os.mkdir(DESTINATION)

dataset_path = os.getcwd() + '/Dataset'
if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
config.DOWNLOADED_DATASETS_PATH = Path(dataset_path)








#############################################################
"""#Reproducibility"""
#############################################################

def set_seed(seed, loader=None):
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

set_seed(CONFIG.seed)








#################################################################
"""# Dataset download and normalize"""
#################################################################

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
print(SYN_DATASET)





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

class CkNmtTokenizer:
    TMP_FOLDER = 'ck_nmt_tokenizer_tmp'

    @staticmethod
    def init():
        if not os.path.isdir(CkNmtTokenizer.TMP_FOLDER):
            os.mkdir(CkNmtTokenizer.TMP_FOLDER)

    @staticmethod
    def save_json(data, path: str) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def createTokenizer(vocab_size):
        print(f'Creating vocab_size: {vocab_size}')

        # optimize/shorter the below algos using python libraries
        source_sentences = []
        ## skip benchmark set

        for sen in RAW_DATASET['parallel'][CONFIG.source_lang]:
            if sen != None:
                source_sentences.append(sen)
        for sen in RAW_DATASET['monolingual'][CONFIG.source_lang]:
            if sen != None:
                source_sentences.append(sen)

        print('Total source sentences' + str(len(source_sentences)))

        target_sentences = []
        ## skip benchmark set

        for sen in RAW_DATASET['parallel'][CONFIG.target_lang]:
            if sen != None:
                target_sentences.append(sen)
        for sen in RAW_DATASET['monolingual'][CONFIG.target_lang]:
            if sen != None:
                target_sentences.append(sen)

        print('Total target sentences:' + str(len(target_sentences)))

        with open(CkNmtTokenizer.TMP_FOLDER + '/sentences.src', 'w+', encoding='utf-8') as df:
            for sen in source_sentences:
                df.write(sen + '\n')

        with open(CkNmtTokenizer.TMP_FOLDER + '/sentences.tgt', 'w+', encoding='utf-8') as df:
            for sen in target_sentences:
                df.write(sen + '\n')

        print('Applying spm...')
        spm.SentencePieceTrainer.train(input=CkNmtTokenizer.TMP_FOLDER + '/sentences.src',
                                       model_prefix=CONFIG.source_lang, vocab_size=vocab_size,
                                       user_defined_symbols=['<pad>'])
        spm.SentencePieceTrainer.train(input=CkNmtTokenizer.TMP_FOLDER + '/sentences.tgt',
                                       model_prefix=CONFIG.target_lang, vocab_size=vocab_size,
                                       user_defined_symbols=['<pad>'])

        source_model = f'./{CONFIG.source_lang}.model'
        target_model = f'./{CONFIG.target_lang}.model'

        sp = spm.SentencePieceProcessor(model_file=source_model)
        vocabs = [[sp.id_to_piece(id), id] for id in range(sp.get_piece_size())]
        vocabDict = {}
        for pair in vocabs:
            vocabDict[pair[0]] = pair[1]
        CkNmtTokenizer.save_json(vocabDict, 'marian_vocab.src')

        sp = spm.SentencePieceProcessor(model_file=target_model)
        vocabs = [[sp.id_to_piece(id), id] for id in range(sp.get_piece_size())]
        vocabDict = {}
        for pair in vocabs:
            vocabDict[pair[0]] = pair[1]
        CkNmtTokenizer.save_json(vocabDict, 'marian_vocab.tgt')

        print('Creating MarianTokenizer...')

        source2target_tokenizer = MarianTokenizer(source_model, target_model, 'marian_vocab.src', 'marian_vocab.tgt',
                                                  CONFIG.source_lang, CONFIG.target_lang, separate_vocabs=True,
                                                  model_max_length=CONFIG.max_length)
        target2source_tokenizer = MarianTokenizer(target_model, source_model, 'marian_vocab.tgt', 'marian_vocab.src',
                                                  CONFIG.target_lang, CONFIG.source_lang, separate_vocabs=True,
                                                  model_max_length=CONFIG.max_length)

        print('MarianTokenizer created!')
        return {'source': source2target_tokenizer, 'target': target2source_tokenizer}


CkNmtTokenizer.init()
TOKENIZER = CkNmtTokenizer.createTokenizer(vocab_size=CONFIG.vocab_size)['source' if CONFIG.is_forward else 'target']










# =====================================================================================
"""""""""""""""""""""""""""""""""""""# Model"""""""""""""""""""""""""""""""""""""""""""
# =====================================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, num_layers ):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers  = num_layers, bidirectional = True)
        self.fc = nn.Linear(dec_hid_dim*num_layers*2, dec_hid_dim*num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        #src = [src len, batch size]
        batch_size = src.shape[1]

        embedded = self.dropout(self.embedding(src)) # [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)

        # change hidden to -> [batch size, dec_hid_dim*num_layers] to make similar with decoder hidden-ouput's dimension (decoder isn't bidirectional)
        hidden = hidden.permute(1, 0, 2) #[batch size, n layers * 2, hid dim]
        hidden = hidden.reshape(batch_size, -1) #[batch size, dec_hid_dim*num_layers*2]
        hidden = torch.tanh(self.fc(hidden))

        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec_hid_dim*num_layers]
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, num_layers):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim*num_layers, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False) # why bias false?

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # [batch size, src_len, dec_hid_dim*num_layers]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch size, src len, enc hid dim * 2]

        # Luong's attention -> concat hidden & encoder_outpus -> then apply linear layer
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) # [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2) #attention= [batch size, src len]

        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, num_layers ):
        super().__init__()

        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dec_hid_dim = dec_hid_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers = num_layers)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):

        batch_size = input.shape[0]

        input = input.unsqueeze(0) # [1, batch size]
        embedded = self.dropout(self.embedding(input)) #  [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs) # [batch size, src len]
        a = a.unsqueeze(1) # [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs) # [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2) # [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim = 2) # [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.reshape(batch_size, self.num_layers, self.dec_hid_dim).permute(1, 0, 2).contiguous())

        output = output.squeeze(0) # [batch_size, dec_hid_dim] - direction & seq_len = 1
        hidden = hidden.permute(1,0,2).reshape(batch_size, -1) # [batch_size, dec_hid_dim * num_layers]

        embedded = embedded.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        top1 = None
        teacher_force = True

        for t in range(trg_len):

            # teacher forcing
            input = trg[t] if teacher_force else top1

            output, hidden = self.decoder(input, hidden, encoder_outputs)

            outputs[t] = output

            top1 = output.argmax(1)

            # deciding teacher_force for next input
            teacher_force = random.random() < teacher_forcing_ratio

        return outputs


    # for Huggingface DataCollatorForSeq2Seq
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.shift_tokens_right(labels, CONFIG.pad_token_id, CONFIG.sos_token_id)

    # for Huggingface DataCollatorForSeq2Seq
    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids






###################################################################
"""# TRANSFORMER INITIALIZATION """
###################################################################

# It is enabled only if the model is a transformer, for working with RNN it is skipped.
TRANSFORMER_MODEL_CONFIG = None
if CONFIG.model_type == 'trans':
    TRANSFORMER_MODEL_CONFIG = MarianConfig(
            vocab_size=CONFIG.vocab_size,
            decoder_vocab_size=CONFIG.vocab_size,
            max_position_embeddings=CONFIG.max_length,
            encoder_layers=CONFIG.trans_layers,
            encoder_ffn_dim=CONFIG.trans_ff_rim,
            encoder_attention_heads=CONFIG.trans_att_heads,
            decoder_layers=CONFIG.trans_layers,
            decoder_ffn_dim=CONFIG.trans_ff_rim,
            decoder_attention_heads=CONFIG.trans_att_heads,
            encoder_layerdrop=CONFIG.trans_layer_dropout,
            decoder_layerdrop=CONFIG.trans_layer_dropout,
            use_cache=True,
            is_encoder_decoder=True,
            activation_function="gelu",
            d_model=CONFIG.trans_d_model,
            dropout=CONFIG.trans_dropout,
            attention_dropout=CONFIG.trans_dropout,
            activation_dropout=CONFIG.trans_actv_dropout,
            init_std=0.02,
            decoder_start_token_id=CONFIG.sos_token_id,
            scale_embedding=False,
            pad_token_id=CONFIG.pad_token_id,
            eos_token_id=CONFIG.eos_token_id,
            forced_eos_token_id=None,
            share_encoder_decoder_embeddings=False
    )

# ==============================================================================================









#####################################################################################
"""# Model setup and Weight Initializations"""
#####################################################################################

def glorot_initialization(m):
    #https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/.
    if hasattr(m, "weight") and m.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(m.weight.data) # also called glorot initialization

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=CONFIG.rnn_weight_init_std)
        else:
            nn.init.constant_(param.data, 0)

def create_model():
    if CONFIG.model_type == 'rnn':
        attention = Attention(CONFIG.rnn_hidden_size, CONFIG.rnn_hidden_size, CONFIG.rnn_enc_dec_num_layer)
        encoder = Encoder(CONFIG.vocab_size, CONFIG.rnn_embedding_size, CONFIG.rnn_hidden_size, CONFIG.rnn_hidden_size, CONFIG.rnn_hidden_embedding_dropout, CONFIG.rnn_enc_dec_num_layer)
        decoder = Decoder(CONFIG.vocab_size, CONFIG.rnn_embedding_size, CONFIG.rnn_hidden_size, CONFIG.rnn_hidden_size, CONFIG.rnn_hidden_embedding_dropout, attention, CONFIG.rnn_enc_dec_num_layer)
        model = Seq2Seq(encoder, decoder, CONFIG.device)
        model.apply(init_weights)
    else:
        model = MarianMTModel(TRANSFORMER_MODEL_CONFIG)
        model.apply(glorot_initialization)

    model = model.to(CONFIG.device) # remove other models from GPU

    optimizer = AdamW(model.parameters(), lr=CONFIG.learning_rate)  # TODO: replace with torch.optim.AdamW ??
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=CONFIG.num_warmup_steps,
        num_training_steps=CONFIG.num_train_steps,
    )

    # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda cur_step: 1.0)

    return model, optimizer, lr_scheduler








##########################################################
"""# Combined train, val, monoglingual data"""
##########################################################

TRAIN_TEST_RAW_DATASETS = DatasetDict({'train': SYN_DATASET['train'], 'validation': DEV_VALIDATION_DATASET['dev_val']})
MONOLINGUAL_DATA = RAW_DATASET['monolingual']

print(TRAIN_TEST_RAW_DATASETS)
print(MONOLINGUAL_DATA)








############################################
"""# Tokenization function"""
############################################

def tokenization(features, tokenizer = None, forward_train = True):
    if forward_train:
        return tokenizer(features[CONFIG.source_lang], text_target=features[CONFIG.target_lang], max_length=CONFIG.max_length, truncation=True)
    else:
        return tokenizer(features[CONFIG.target_lang], text_target=features[CONFIG.source_lang], max_length=CONFIG.max_length, truncation=True)









############################################
"""# DataLoader"""
############################################

def create_dataloader(tokenized_dataset, data_collator):
    train_dataset = tokenized_dataset['train']
    validation_dataset= tokenized_dataset['validation']

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=CONFIG.train_batch_size,
    )
    eval_dataloader = DataLoader(
        validation_dataset,
        collate_fn=data_collator,
        batch_size=CONFIG.validation_batch_size
    )

    return train_dataloader, eval_dataloader

def prepare_dataloader_for_training(tokenizer, data_collator, forward_train):
    # tokenize dataset
    tokenized_dataset = TRAIN_TEST_RAW_DATASETS.map(
        partial(tokenization, tokenizer = tokenizer, forward_train = forward_train),
        batched=True,
        remove_columns=TRAIN_TEST_RAW_DATASETS['train'].column_names,
    )

    tokenized_dataset.set_format("torch")

    # dataloader
    return create_dataloader(tokenized_dataset, data_collator)

def prepare_dataloader_for_benchmark(tokenizer, data_collator, forward_train):
    # tokenize dataset
    tokenized_dataset = BENCHMARK_DATASET.map(
        partial(tokenization, tokenizer = tokenizer, forward_train = forward_train),
        batched=True,
        remove_columns=BENCHMARK_DATASET.column_names,
    )

    tokenized_dataset.set_format("torch")

    # dataloader
    return DataLoader(
        tokenized_dataset,
        collate_fn=data_collator,
        batch_size=CONFIG.validation_batch_size,
    )









#########################################################
"""#Custom Beam Decoder"""
#########################################################

#########################################################
## Beam Search (For single sample only, not batch)
############################################################################################
## Idea is to create a node with parents for each decoded output, keep them in priorityqueue
## then expore node with tNUM_TRAIN_EPOCHShe minimum error(top node of the priorityqueue)
## we can create the sentences by backtracking the node to its parent nodes
############################################################################################

class Beam_Limit_Counter:
    value=0

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        #https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/issues/3
        return self.logp < other.logp




def beam_decode(decoder, decoder_hidden, encoder_outputs, tokenizer):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch

    here batch size is typically 1
    '''

    beam_width = CONFIG.beam_width
    topk = CONFIG.top_k  # how many sentence do you want to generate

    # Start with the start of the sentence token
    decoder_input = torch.tensor([CONFIG.sos_token_id], device=CONFIG.device)  # SOS

    # Number of sentence to generate
    endnodes = []
    number_required = CONFIG.beam_endpoints

    # starting node -  hidden vector, previous node, word id, logp, length
    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > CONFIG.queue_limit:
            Beam_Limit_Counter.value += 1
            break

        # fetch the best node
        score, n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h

        if n.wordid.item() == CONFIG.eos_token_id and n.prevNode != None:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        # not storing decoder attention for now

        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        # decoder_output = [batch size = 1, output dim]
        # decoder_hidden =  [batch_size, dec_hid_dim * num_layers]

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = torch.topk(decoder_output, beam_width)
        nextnodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[0][new_k].view(1)
            log_p = log_prob[0][new_k].item()

            node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))
            # increase qsize
        qsize += len(nextnodes) - 1

    # choose nbest paths, back trace them

    utterances = []
    for idx, (score, n) in enumerate(sorted(endnodes, key=operator.itemgetter(0))):
        if idx == topk:
            break
        utterance = []
        utterance.append(n.wordid.item())

        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid.item())

        utterance = utterance[::-1]
        sentence = tokenizer.decode(utterance, skip_special_tokens=True)
        utterances.append(sentence)

    if len(utterances) == 0:
        score, n = nodes.get()

        utterance = []
        utterance.append(n.wordid.item())

        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid.item())

        utterance = utterance[::-1]
        sentence = tokenizer.decode(utterance, skip_special_tokens=True)
        utterances.append(sentence)

    return utterances




def evaluateWithBeamSearch(encoder, decoder, tokenizer, sentence, max_length=CONFIG.max_length):
    with torch.no_grad():
        sentence = normalize(sentence)
        token_data = tokenizer(sentence, return_tensors= 'pt', max_length=CONFIG.max_length, truncation=True)

        token_data = {k: v.to(CONFIG.device) for k, v in token_data.items()}

        inputs = token_data['input_ids'].permute(1, 0) # [input_len, batch_size]

        encoder_outputs, encoder_hidden = encoder(inputs)
        translations = beam_decode(decoder, encoder_hidden, encoder_outputs, tokenizer)

        return translations, None







##############################################################
"""# Benchmark Testing Function"""
##############################################################

def test_benchmark(model, data_loader, tokenizer):
    loss_fct_bk = CrossEntropyLoss(label_smoothing = CONFIG.label_smoothing)

    eval_lossesss = 0
    eval_total_samples = 0

    model.eval()

    for batch in data_loader:
        batch = {k: v.to(CONFIG.device) for k, v in batch.items()}
        eval_total_samples += batch['input_ids'].shape[0]

        if CONFIG.model_type == 'rnn':
            inputs = batch['input_ids'].permute(1, 0)  # [input_len, batch_size]
            decoder_input_ids = batch['decoder_input_ids'].permute(1, 0)  # [targer_len, batch_size]
            labels = batch['labels']  # [batch_size, target_len]
            with torch.no_grad():
                logits = model(inputs, decoder_input_ids, 0)  # [target_len, batch_size, vocab_size]

            ls_loss = loss_fct_bk(logits.reshape(-1, CONFIG.vocab_size), labels.permute(1, 0).reshape(-1))
            predictions = torch.argmax(logits, dim=-1).permute(1, 0)  # [batch_size, target_len]
            decoded_preds, decoded_labels = postprocess(predictions, labels, tokenizer)
        else:
            labels = batch['labels']
            trans_outputs = model(**batch)
            logits = trans_outputs.logits
            ls_loss = loss_fct_bk(logits.view(-1, model.config.decoder_vocab_size), labels.view(-1))

            predictions = torch.argmax(logits, dim=-1)
            decoded_preds, decoded_labels = postprocess(predictions, batch["labels"], tokenizer)


        eval_lossesss += ls_loss.item()

        bleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        chrf_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        meteor_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    bleu_resultss = bleu_metric.compute(force=True)
    chrf_resultss = chrf_metric.compute()
    meteor_resultss = meteor_metric.compute()

    print(f"Benchmark results: LR: Loss {(eval_lossesss / eval_total_samples):.2f}, BLEU score: {bleu_resultss['score']:.2f}, CHRF score: {chrf_resultss['score']:.2f}, METEOR score: {meteor_resultss['meteor']:.2f}")








##############################################################
"""# Train Function"""
##############################################################

def train_and_validate(train_dataloader, eval_dataloader, benchmark_dataloader, model, optimizer, lr_scheduler, tokenizer, run):
    print(f'Training started! RUN:{run}')

    progress_bar = tqdm(range(CONFIG.num_train_steps))

    direction_prefix = 'source2target' if CONFIG.is_forward == True else 'target2source'
    model_prefix = direction_prefix + '-run' + str(run)

    loss_fct = CrossEntropyLoss(label_smoothing = CONFIG.label_smoothing)

    performance_dict = dict()
    performance_dict['epoch'] = []
    performance_dict['last_lr'] = []
    performance_dict['train_loss'] = []
    performance_dict['train_bleu'] = []
    performance_dict['eval_loss'] = []
    performance_dict['eval_bleu'] = []
    performance_dict['eval_chrf'] = []
    performance_dict['eval_meteor'] = []

    best_score = dict()
    best_score['bleu'] = 0
    best_score['meteor'] = 0
    best_score['chrf'] = 0

    current_early_stoppings = CONFIG.early_stoppings
    current_steps = 0

    model_dict = dict()

    for epoch in range(CONFIG.num_train_epochs):

        model.train()
        lossess = 0
        total_samples = 0

        for batch in train_dataloader:
            current_steps += 1

            batch = {k: v.to(CONFIG.device) for k, v in batch.items()}
            total_samples += batch['input_ids'].shape[0]

            if CONFIG.model_type == 'rnn':
                inputs = batch['input_ids'].permute(1, 0)  # [input_len, batch_size]
                decoder_input_ids = batch['decoder_input_ids'].permute(1, 0)  # [targer_len, batch_size]
                labels = batch['labels'].permute(1, 0)  # [target_len, batch_size]
                logits = model(inputs, decoder_input_ids)  # [target_len, batch_size, vocab_size]
                ls_loss = loss_fct(logits.reshape(-1, CONFIG.vocab_size), labels.reshape(-1))

                predictions = torch.argmax(logits, dim=-1).permute(1, 0)  # [batch_size, target_len]
                decoded_preds, decoded_labels = postprocess(predictions, labels.permute(1, 0), tokenizer)
            else:
                labels = batch['labels']
                trans_outputs = model(**batch)
                logits = trans_outputs.logits
                ls_loss = loss_fct(logits.view(-1, model.config.decoder_vocab_size), labels.view(-1))

                predictions = torch.argmax(logits, dim=-1)
                decoded_preds, decoded_labels = postprocess(predictions, batch["labels"], tokenizer)

            lossess += ls_loss.item()
            ls_loss.backward()

            # gradient clipping
            if CONFIG.clip_grad != None:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=CONFIG.clip_grad)

            optimizer.step()
            lr_scheduler.step()  # of no use since lr is constant
            optimizer.zero_grad()
            progress_bar.update(1)

            train_bleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        eval_lossess = 0
        eval_total_samples = 0
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(CONFIG.device) for k, v in batch.items()}
            eval_total_samples += batch['input_ids'].shape[0]

            if CONFIG.model_type == 'rnn':
                inputs = batch['input_ids'].permute(1, 0)  # [input_len, batch_size]
                decoder_input_ids = batch['decoder_input_ids'].permute(1, 0)  # [targer_len, batch_size]
                labels = batch['labels']  # [batch_size, target_len]
                with torch.no_grad():
                    logits = model(inputs, decoder_input_ids, 0)  # [target_len, batch_size, vocab_size]

                ls_loss = loss_fct(logits.reshape(-1, CONFIG.vocab_size), labels.permute(1, 0).reshape(-1))
                predictions = torch.argmax(logits, dim=-1).permute(1, 0)  # [batch_size, target_len]
                decoded_preds, decoded_labels = postprocess(predictions, labels, tokenizer)
            else:
                labels = batch['labels']
                trans_outputs = model(**batch)
                logits = trans_outputs.logits
                ls_loss = loss_fct(logits.view(-1, model.config.decoder_vocab_size), labels.view(-1))

                predictions = torch.argmax(logits, dim=-1)
                decoded_preds, decoded_labels = postprocess(predictions, batch["labels"], tokenizer)

            eval_lossess += ls_loss.item()

            bleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            chrf_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            meteor_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        bleu_results = bleu_metric.compute(force=True)
        train_bleu_results = train_bleu_metric.compute(force=True)
        chrf_results = chrf_metric.compute()
        meteor_results = meteor_metric.compute()

        performance_dict['epoch'].append(epoch)
        performance_dict['last_lr'].append(lr_scheduler.get_last_lr())
        performance_dict['train_loss'].append(lossess / total_samples)
        performance_dict['train_bleu'].append(train_bleu_results['score'])
        performance_dict['eval_loss'].append(eval_lossess / eval_total_samples)
        performance_dict['eval_bleu'].append(bleu_results['score'])
        performance_dict['eval_chrf'].append(chrf_results['score'])
        performance_dict['eval_meteor'].append(meteor_results['meteor'])

        # ## reduce on plateau
        # lr_scheduler.step(bleu_results['score'])

        print(f"epoch {epoch}, LR: {lr_scheduler.get_last_lr()}, Train Loss {(lossess / total_samples):.2f}, Train Bleu {(train_bleu_results['score']):.2f}, Eval Loss {(eval_lossess / eval_total_samples):.2f}, BLEU score: {bleu_results['score']:.2f}, CHRF score: {chrf_results['score']:.2f}, METEOR score: {meteor_results['meteor']:.2f}")

        improved = False
        improvement_msg = ''
        if best_score['bleu'] < bleu_results['score']:
            improved = True
            improvement_msg += 'bleu...'
            best_score['bleu'] = bleu_results['score']

        if best_score['chrf'] < chrf_results['score']:
            improved = True
            improvement_msg += 'chrf...'
            best_score['chrf'] = chrf_results['score']

        if best_score['meteor'] < meteor_results['meteor']:
            improved = True
            improvement_msg += 'meteor...'
            best_score['meteor'] = meteor_results['meteor']

        ## benchmark testing
        test_benchmark(model, benchmark_dataloader, tokenizer)

        if not improved:
            print('**** No improvements!!! ... early stoppings count reduced. ****')
            current_early_stoppings -= 1

            if current_early_stoppings <= 0:
                print('Training complete!')
                break
        else:
            print(improvement_msg + '  improved!!')

            current_model_path = DESTINATION + model_prefix + '-epoch' + str(epoch) + '.pt'

            torch.save({'epoch': epoch,
                        'model': model,
                        # 'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                        }, current_model_path)

            print(f'Saving model...: {current_model_path}')

            model_dict[epoch] = current_model_path

            if len(model_dict) > CONFIG.keep_top_models:
                model_key = sorted(model_dict.keys())[0]

                os.remove(model_dict[model_key])
                print(f'Removing bottom model...: {model_dict[model_key]}')

                model_dict.pop(model_key)

        if current_steps >= CONFIG.num_train_steps:
            print('Total Train steps done! Stopping training.')
            break

    # with open(DESTINATION + "performance_dict-" + model_prefix + ".json", "w") as f:
    #     json.dump(performance_dict, f)
    # convert to int

    return performance_dict







##############################################
"""# Metrics"""
##############################################

bleu_metric = evaluate.load("sacrebleu")
train_bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
meteor_metric = evaluate.load("meteor")






#######################################################
"""# Post Process Method for Score Calculation"""
#######################################################

def postprocess(predictions, labels, tokenizer):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces = False)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces = False)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels






##################################################################
"""# Synthetic Data Generation"""
##################################################################

def generate_synthetic_dataset(forward_train, model, tokenizer):
    if forward_train:
        input_data = MONOLINGUAL_DATA[CONFIG.source_lang]
    else:
        input_data = MONOLINGUAL_DATA[CONFIG.target_lang]

    encoder = None
    decoder = None
    if CONFIG.model_type == "rnn":
        encoder = model.encoder
        decoder = model.decoder

    outputs = []
    raw_inputs = []
    with torch.no_grad():
        syn_cnt = 0
        Beam_Limit_Counter.value = 0

        for sentence in input_data:
            if sentence == None or syn_cnt == CONFIG.total_syn_to_generate:
                break

            sentence = normalize(sentence)
            raw_inputs.append(sentence)

            token_data = tokenizer(sentence, return_tensors= 'pt', max_length=CONFIG.max_length, truncation=True)
            token_data = {k: v.to(CONFIG.device) for k, v in token_data.items()}



            if CONFIG.model_type == 'rnn':
                inputs = token_data['input_ids'].permute(1, 0) # [input_len, batch_size]
                encoder_outputs, encoder_hidden = encoder(inputs)
                translations = beam_decode(decoder, encoder_hidden, encoder_outputs, tokenizer) # ********************************************* check if special tokens needed to be removed
            else:
                generated_ids = model.generate(**token_data, num_beams=CONFIG.beam_width, max_length=CONFIG.max_length, num_return_sequences=CONFIG.top_k)
                translations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            outputs.append(translations[0]) # top translation
            torch.cuda.empty_cache()
            syn_cnt += 1

            if syn_cnt % 100 == 0:
                print(f'\r{syn_cnt}/{CONFIG.total_syn_to_generate}, nodes limit crossed: {Beam_Limit_Counter.value}', end = '')

    if forward_train:
        return {CONFIG.source_lang: raw_inputs, CONFIG.target_lang: outputs}
    else:
        return {CONFIG.source_lang: outputs, CONFIG.target_lang: raw_inputs} # we generated source

def upload_synthetic_data(syn_new_data):
    ## create data
    parallelDataset = Dataset.from_dict({
                                        CONFIG.source_lang: BASE_SYN['train'][CONFIG.source_lang] + syn_new_data[CONFIG.source_lang],
                                        CONFIG.target_lang:  BASE_SYN['train'][CONFIG.target_lang] + syn_new_data[CONFIG.target_lang]
                                        })

    ## create dataset
    combined_new_syn_dataset = DatasetDict({'train':  parallelDataset})

    ## upload
    combined_new_syn_dataset.push_to_hub(CONFIG.synthetic_dataset_next)







########################################################
"""# Main Steps"""
########################################################

# create model and others
model, optimizer, lr_scheduler = create_model()

data_collator = DataCollatorForSeq2Seq(TOKENIZER, model=model)

# prepare dataloader
train_dataloader, eval_dataloader = prepare_dataloader_for_training(TOKENIZER, data_collator, CONFIG.is_forward)
benchmark_dataloader = prepare_dataloader_for_benchmark(TOKENIZER, data_collator, CONFIG.is_forward)

# train
results = train_and_validate(train_dataloader, eval_dataloader, benchmark_dataloader, model, optimizer, lr_scheduler, TOKENIZER, CONFIG.run_id)






########################################################################################################
"""# Generating Synthetic Data on The Monolingual Set and Uploading to Huggingface Repository """
########################################################################################################

# #How to check the best model?
# best_path = DESTINATION + 'target2source-run0-epoch18.pt'

# # select best model
# checkpoint  = torch.load(best_path)
# best_model = checkpoint['model']

# # # #Huggingface login
# notebook_login()

# # # generate synthetic data and upload # ~2hr for 42700 # 32mins for 42700
# gen_syn_data = generate_synthetic_dataset(CONFIG.is_forward, best_model, TOKENIZER)

# upload_synthetic_data(gen_syn_data)







#################################################################################
"""# Benchmark Testing"""
#################################################################################

# # # select best model
# best_path = DESTINATION + 'target2source-run1-epoch14.pt'
# checkpoint  = torch.load(best_path)
# best_model = checkpoint['model']

# data_collator = DataCollatorForSeq2Seq(TOKENIZER, model=best_model)
# benchmark_dataloader = prepare_dataloader_for_benchmark(TOKENIZER, data_collator, CONFIG.is_forward)
# train_dataloader, eval_dataloader = prepare_dataloader_for_training(TOKENIZER, data_collator, CONFIG.is_forward)

# test_benchmark(best_model, benchmark_dataloader, TOKENIZER)

# test_benchmark(best_model, eval_dataloader, TOKENIZER)






###########################################################################
"""# Single Sentence Predition """
###########################################################################

# Please tweak it according to your need
# # select best model
# best_path = './cl-nmt-training-models/2023-09-29-v1/s-run0-epoch11.pt'
# checkpoint  = torch.load(best_path)
# best_model = checkpoint['model']

# model = best_model

# # sen = 'আমাদের সৃষ্টিকর্তা , যিহোবা সঙ্গে সঙ্গেই সেই বিদ্রোহের কারণে উদ্ভূত বিভিন্ন ঘটনার মন্দ প্রভাবগুলোকে দূর করার জন্য একটা ব্যবস্থা নির্ধারণ করেছিলেন ।'
# sen = '𑄚𑄧 𑄛𑄢𑄨𑄟𑄴'
# sen = normalize(sen)
# print(sen)
# print(TOKENIZER.tokenize(sen))
# token_data = TOKENIZER(sen, return_tensors= 'pt')
# token_data = {k: v.to(CONFIG.device) for k, v in token_data.items()}

# inputs = token_data['input_ids'].permute(1, 0) # [input_len, batch_size]

# print(token_data)

# encoder_outputs, encoder_hidden = model.encoder(inputs)
# # print(encoder_outputs)

# translations = beam_decode(model.decoder, encoder_hidden, encoder_outputs, TOKENIZER)

# print(translations)

