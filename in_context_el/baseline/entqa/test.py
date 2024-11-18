import os
import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer
import argparse

import_path = '/nfs/yding4/EntQA'
sys.path.insert(0, import_path)
from gerbil_experiments.nn_processing import Annotator

'''
from utils import Logger
from run_retriever import load_model as load_retriever
from run_reader import load_model as load_reader
from data_retriever import (
    get_embeddings,
    get_hard_negative,
)
from reader import prune_predicts
from gerbil_experiments.data import (
    get_retriever_loader, get_reader_loader, 
    load_entities, 
    get_reader_input, process_raw_data,
    get_doc_level_predicts, token_span_to_gerbil_span,
    get_raw_results, process_raw_predicts,
)
'''


if __name__ == '__main__':

    '''
    # https://github.com/WenzhengZhang/EntQA/blob/main/gerbil_experiments/server.py#L71
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str,
                        help='log path')
    parser.add_argument('--blink_dir', type=str,
                        help='blink pretrained bi-encoder path')
    parser.add_argument(
        "--passage_len", type=int, default=32,
        help="the length of each passage"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="length of stride when chunking passages",
    )
    parser.add_argument('--bsz_retriever', type=int, default=4096,
                        help='the batch size of retriever')
    parser.add_argument('--max_len_retriever', type=int, default=42,
                        help='max length of the retriever input passage ')
    parser.add_argument('--retriever_path', type=str,
                        help='trained retriever path')
    parser.add_argument('--type_retriever_loss', type=str,
                        default='sum_log_nce',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='type of marginalize for retriever')
    parser.add_argument('--gpus', default='', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--cands_embeds_path', type=str,
                        help='the path of candidates embeddings')
    parser.add_argument('--k', type=int, default=100,
                        help='top-k candidates for retriever')
    parser.add_argument('--ents_path', type=str,
                        help='entity file path')
    parser.add_argument('--max_len_reader', type=int, default=180,
                        help='max length of joint input [%(default)d]')
    parser.add_argument('--max_num_candidates', type=int, default=100,
                        help='max number of candidates [%(default)d] when '
                             'eval for reader')
    parser.add_argument('--bsz_reader', type=int, default=32,
                        help='batch size [%(default)d]')
    parser.add_argument('--reader_path', type=str,
                        help='trained reader path')
    parser.add_argument('--type_encoder', type=str,
                        default='squad2_electra_large',
                        help='the type of encoder')
    parser.add_argument('--type_span_loss', type=str,
                        default='sum_log',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='the type of marginalization for reader')
    parser.add_argument('--type_rank_loss', type=str,
                        default='sum_log',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='the type of marginalization for reader')
    parser.add_argument('--num_spans', type=int, default=3,
                        help='top num_spans for evaluation on reader')
    parser.add_argument('--thresd', type=float, default=0.05,
                        help='probabilty threshold for evaluation on reader')
    parser.add_argument('--max_answer_len', type=int, default=10,
                        help='max length of answer [%(default)d]')
    parser.add_argument('--max_passage_len', type=int, default=32,
                        help='max length of question [%(default)d] for reader')
    parser.add_argument('--document', type=str,
                        help='test document')
    parser.add_argument('--save_span_path', type=str,
                        help='save span-based document-level results path')
    parser.add_argument('--save_char_path', type=str,
                        help='save char-based path')
    parser.add_argument('--add_topic', action='store_true',
                        help='add title?')
    parser.add_argument('--do_rerank', action='store_true',
                        help='do reranking for reader?')
    parser.add_argument('--use_title', action='store_true',
                        help='use title?')
    parser.add_argument('--no_multi_ents', action='store_true',
                        help='no repeated entities are allowed given a span?')
    '''

    entqa_data_dir = '/nfs/yding4/EntQA/data/'
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

    entqa_config = argparse.Namespace(**config)
    annotator = Annotator(entqa_config)

    sentence = "England won the FIFA World Cup in 1966."
    annotator.get_predicts(sentence)

