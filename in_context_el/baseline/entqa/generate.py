import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm


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

def entqa4el(sentence, spans, annotator, return_ori=False):
    output = annotator.get_predicts(sentence)
    if return_ori:
        return output

    starts = []
    ends = []
    entity_mentions = []
    entity_names = []
    for span in output:
        assert len(span) == 3
        start, l, entity_name = span
        starts.append(start)
        ends.append(start + l)
        entity_mentions.append(sentence[start: start + l])
        entity_names.append(entity_name)

    pred_entities = {
        'starts': starts,
        'ends': ends,
        'entity_mentions': entity_mentions,
        'entity_names': entity_names,
    }
    return pred_entities


def parse_args():
    parser = argparse.ArgumentParser(
        description='argument for use RefinED for ED/EL',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_dir",
        help="the processed dataset file",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/datasets/",
        type=str,
    )
    parser.add_argument(
        "--datasets",
        help="the processed dataset file",
        # required=True,
        default="['KORE50','msnbc','oke_2015','oke_2016','Reuters-128','RSS-500','aida_test','derczynski']",
        type=eval,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/baseline/entqa/EL_standard_datasets/prediction",
        type=str,
    )
    parser.add_argument(
        "--entqa_dir",
        help="model parameter",
        default="/nfs/yding4/EntQA",
        # required=True,
        type=str,
    )


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    assert os.path.isdir(args.input_dir)
    return args


def predict_el():
    args = parse_args()
    sys.path.insert(0, args.entqa_dir)
    from gerbil_experiments.nn_processing import Annotator
    entqa_data_dir = os.path.join(args.entqa_dir, 'data/')

    # https://github.com/WenzhengZhang/EntQA/blob/main/gerbil_experiments/server.py#L71
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

    for dataset in args.datasets:
        print(f'dataset: {dataset}')
        input_file = os.path.join(args.input_dir, dataset + '.json')
        output_file = os.path.join(args.output_dir, dataset + '.json')
            
        with open(input_file) as reader:
            doc_name2instance = json.load(reader)

        if os.path.isfile(output_file):
            with open(output_file) as reader:
                doc_name2instance = json.load(reader)

        for doc_name, instance in tqdm(doc_name2instance.items()):
            if 'pred_entities' in instance:
                continue
            sentence = instance['sentence']
            entities = instance['entities']

            span_list = []
            for (
                start,
                end,
            ) in zip(
                entities['starts'],
                entities['ends'],
            ):
                span = (start, end)
                span_list.append(span)

            pred_entities = entqa4el(
                    sentence, 
                    span_list,
                    annotator,
                    return_ori=False,
            )

            doc_name2instance[doc_name]['pred_entities'] = pred_entities

            with open(output_file, 'w') as writer:
                json.dump(doc_name2instance, writer, indent=4)


if __name__ == '__main__':
    predict_el()
