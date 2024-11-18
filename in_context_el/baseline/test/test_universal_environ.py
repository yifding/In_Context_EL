import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import argparse


def test_blink(sentence, spans):
    import blink.main_dense as main_dense
    import torch
    torch.cuda.set_device(1)

    from in_context_el.baseline.blink.generate import blink4ed

    blink_num_candidates = 100
    models_path = "/nfs/yding4/EL_project/BLINK/models/"

    config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": blink_num_candidates,
            "biencoder_model": models_path+"biencoder_wiki_large.bin",
            "biencoder_config": models_path+"biencoder_wiki_large.json",
            "entity_catalogue": models_path+"entity.jsonl",
            "entity_encoding": models_path+"all_entities_large.t7",
            "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
            "crossencoder_config": models_path+"crossencoder_wiki_large.json",
            "fast": False, # set this to be true if speed is a concern
            "output_path": "logs/" # logging directory
    }

    blink_args = argparse.Namespace(**config)

    models = main_dense.load_models(blink_args, logger=None)

    (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = models

    output = blink4ed(
        sentence, 
        spans, 
        title2id,   # from blink
        id2text,    # from blink
        blink_args, # from blink
        main_dense, # from blink
        models, # from blink
        k=10, # for blink candidates
        num_context_characters=150, 
        num_entity_description_characters=150,
        return_ori=False,
    )

    print(json.dumps(output, indent=4))


def test_refined(sentence, spans):
    from refined.inference.processor import Refined
    from refined.data_types.base_types import Span
    from in_context_el.baseline.refined.generate import refined4ed_el
    model_name = "aida_model"
    entity_set = "wikipedia"
    refined = Refined.from_pretrained(
        model_name=model_name,
        entity_set=entity_set,
        use_precomputed_descriptions=True,
    )

    print('test RefinED for entity disambiguation: \n')
    output = refined4ed_el(sentence, spans, refined, el=False, return_ori=False)
    print(json.dumps(output, indent=4))

    print('test RefinED for entity linking: \n')
    output = refined4ed_el(sentence, spans, refined, el=True, return_ori=False)
    print(json.dumps(output, indent=4))


def test_entqa(sentence, spans):
    from transformers import BertTokenizer
    import_path = '/nfs/yding4/EntQA'
    sys.path.insert(0, import_path)
    from gerbil_experiments.nn_processing import Annotator

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

    # sentence = "England won the FIFA World Cup in 1966."
    annotator.get_predicts(sentence)


if __name__ == '__main__':
    sentence = "England won the FIFA World Cup in 1966."
    spans = [
        (0, 7),
        (16, 30),
    ]
    # test_blink(sentence, spans)
    test_refined(sentence, spans)

