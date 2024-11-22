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

    blink_num_candidates = 10
    models_path = "/afs/crc.nd.edu/user/y/yding4/EL_project/BLINK/models/"

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
    import_path = '/scratch365/yding4/EntQA'
    sys.path.insert(0, import_path)
    from gerbil_experiments.nn_processing import Annotator

    entqa_data_dir = os.path.join(import_path, 'data')
    config = {
        'device': 'cpu',
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
    output = annotator.get_predicts(sentence)
    print(output)
    return annotator


def test_REL(sentence, spans):

    from REL.mention_detection import MentionDetection
    from REL.utils import process_results
    from REL.entity_disambiguation import EntityDisambiguation
    from REL.ner import Cmns, load_flair_ner

    wiki_version = "wiki_2014"
    base_url = "/nfs/yding4/REL/data/"

    input_text = {
        "my_doc": (sentence, []),
    }

    mention_detection = MentionDetection(base_url, wiki_version)
    tagger_ner = load_flair_ner("ner-fast")

    tagger_ngram = Cmns(base_url, wiki_version, n=5)
    mentions, n_mentions = mention_detection.find_mentions(input_text, tagger_ngram)

    config = {
        "mode": "eval",
        "model_path": "ed-wiki-2014",
    }
    model = EntityDisambiguation(base_url, wiki_version, config)

    predictions, timing = model.predict(mentions)
    result = process_results(mentions, predictions, input_text)
    print(result)
    # {'my_doc': [(0, 13, 'Hello, world!', 'Hello_world_program', 0.6534378618767961, 182, '#NGRAM#')]}


    '''
    from REL.wikipedia import Wikipedia
    from REL.wikipedia_yago_freq import WikipediaYagoFreq
    from REL.mention_detection import MentionDetection
    base_url = "/nfs/yding4/REL/data/"
    wiki_version = "wiki_2014"
    mention_detection = MentionDetection(base_url, wiki_version)
    wikipedia = Wikipedia(base_url, wiki_version)
    wikipedia_yago_freq = WikipediaYagoFreq(base_url, wiki_version, wikipedia)
    wikipedia_yago_freq.extract_entity_description()
    '''


def entqa_custom(sentence, annotator):
    from gerbil_experiments.data import get_retriever_loader, get_reader_loader, \
        load_entities, \
        get_reader_input, process_raw_data, \
        get_doc_level_predicts, token_span_to_gerbil_span, \
        get_raw_results, process_raw_predicts

    from data_retriever import get_embeddings, get_hard_negative
    from reader import prune_predicts

    samples_retriever, token2char_start, token2char_end = process_raw_data(
        sentence,
        annotator.tokenizer,
        annotator.args.passage_len,
        annotator.args.stride
    )

    retriever_loader = get_retriever_loader(
        samples_retriever,
        annotator.tokenizer,
        annotator.args.bsz_retriever,
        annotator.args.max_len_retriever,
        annotator.args.add_topic,
        annotator.args.use_title
    )

    test_mention_embeds = get_embeddings(
        retriever_loader,
        annotator.model_retriever,
        True,
        annotator.my_device,
    )

    top_k_test, scores_k_test = get_hard_negative(
        test_mention_embeds,
        annotator.all_cands_embeds,
        annotator.args.k,
        0,
        False
    )

    samples_reader = get_reader_input(
        samples_retriever,
        top_k_test,
        annotator.entities,
    )

    reader_loader = get_reader_loader(
        samples_reader,
        annotator.tokenizer,
        annotator.args.max_len_reader,
        annotator.args.max_num_candidates,
        annotator.args.bsz_reader,
        annotator.args.add_topic,
        annotator.args.use_title,
    )

    raw_predicts = get_raw_results(
        annotator.model_reader,
        annotator.my_device,
        reader_loader,
        annotator.args.num_spans,
        samples_reader,
        annotator.args.do_rerank,
        True,
        annotator.args.no_multi_ents
    )

    # pruned_predicts = prune_predicts(raw_predicts, annotator.args.thresd)
    pruned_predicts = prune_predicts(raw_predicts, 1e-5)

    transformed_predicts = process_raw_predicts(pruned_predicts,
                                                samples_reader)

    doc_predicts_span = get_doc_level_predicts(transformed_predicts,
                                               annotator.args.stride)

    print(f'token2char_start: {token2char_start}')
    print(f'token2char_end: {token2char_end}')
    print(f'doc_predicts_span: {doc_predicts_span}')
    doc_predicts_gerbil = token_span_to_gerbil_span(doc_predicts_span,
                                                    token2char_start,
                                                    token2char_end)
    return doc_predicts_gerbil

if __name__ == '__main__':
    # sentence = "England won the FIFA World Cup in 1966."
    # spans = [
    #     (0, 7),
    #     (16, 30),
    # ]

    sentence = "Allen founded the EMP in Seattle , which featured exhibitions about Hendrix and Dylan , but also about various science fiction movies ."
    spans = [
        [0, 5],
        [18, 21],
        [25, 32],
        [68, 75],
        [80, 85],
    ]
    # test_blink(sentence, spans)
    # test_refined(sentence, spans)
    # test_REL(sentence, spans)
    # annotator = test_entqa(sentence, spans)

    # debugging the entqa stuff
    output = entqa_custom(sentence, annotator)