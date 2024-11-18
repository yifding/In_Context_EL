import os
import sys
import argparse
import torch


entqa_dir = '/nfs/yding4/EntQA'
sys.path.insert(0, entqa_dir)

from run_retriever import load_model as load_retriever
from run_reader import load_model as load_reader

entqa_data_dir = os.path.join(entqa_dir, 'data')
config = {
    'retriever_path': os.path.join(entqa_data_dir, 'retriever.pt'),
    'cands_embeds_path': os.path.join(entqa_data_dir, 'candidate_embeds.npy'),
    'ents_path': os.path.join(entqa_data_dir, 'kb/entities_kilt.json'),
    'reader_path': os.path.join(entqa_data_dir, 'reader.pt'),
    'blink_dir': os.path.join(entqa_data_dir, 'blink_model/models/'),   # ends with '/'
    'log_path': os.path.join(entqa_data_dir, 'log.txt'),
    # 'save_span_path': '',
    # 'save_char_path': '',
    # 'document': '',
    'bsz_retriever': 128,
    'max_len_retriever': 42,
    'max_len_reader': 180,
    'max_num_candidates': 100,
    'num_spans': 3,
    'bsz_reader': 32,
    'gpus': '',
    'type_encoder': 'squad2_electra_large',
    'k': 100,
    'max_answer_len': 10,
    'max_passage_len': 32,
    'passage_len': 32,
    'stride': 16,
    'type_retriever_loss': 'sum_log_nce',
    'type_rank_loss': 'sum_log',
    'type_span_loss': 'sum_log',
    'thresd': 0.15,
    'add_topic': True,
    'do_rerank': True,
    'use_title': False,
    'no_multi_ents': False,
}

entqa_args = argparse.Namespace(**config)


config = {
    "top_k": 100,
    "biencoder_model": entqa_args.blink_dir + "biencoder_wiki_large.bin",
    "biencoder_config": entqa_args.blink_dir + "biencoder_wiki_large.json"
}

my_device = torch.device('cpu')

model_retriever = load_retriever(
    False,
    config['biencoder_config'],
    entqa_args.retriever_path,
    my_device,
    entqa_args.type_retriever_loss,
    True,
)

print(f'Total number of parameters: {sum(p.numel() for p in model_retriever.parameters())}')

model_reader = load_reader(
    False, 
    entqa_args.reader_path,
    entqa_args.type_encoder,
    my_device,
    entqa_args.type_span_loss,
    entqa_args.do_rerank,
    entqa_args.type_rank_loss,
    entqa_args.max_answer_len,
    entqa_args.max_passage_len
)

print(f'Total number of parameters: {sum(p.numel() for p in model_reader.parameters())}')