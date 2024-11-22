import os
import argparse
from REL.wikipedia import Wikipedia
from REL.wikipedia_yago_freq import WikipediaYagoFreq
from REL.mention_detection import MentionDetection


def rel_entity_candidates_descriptions(mention, rel_args):
    processed_mention = rel_args.mention_detection.preprocess_mention(mention)
    rel_candidate_entities = rel_args.mention_detection.get_candidates(processed_mention)
    rel_processed_candidate_entities = [rel_args.wikipedia.preprocess_ent_name(rel_candidate_entity[0]) for rel_candidate_entity
                                        in rel_candidate_entities[:rel_args.k]]
    descriptions = []
    for rel_processed_candidate_entity in rel_processed_candidate_entities:
        id = wikipedia.ent_wiki_id_from_name(rel_processed_candidate_entity)
        if id in rel_args.wikipedia_yago_freq.entity_id2description:
            description = rel_args.wikipedia_yago_freq.entity_id2description[id][:rel_args.num_entity_description_characters]
        else:
            description = ""
        descriptions.append(description)
    return rel_processed_candidate_entities, descriptions


def blink_entity_candidates_descriptions(mention, left_context, right_context, blink_args):
    data_to_link = [{
        "id": 0,
        "label": "unknown",
        "label_id": -1,
        "context_left": left_context,
        "mention": mention,
        "context_right": right_context,
    }]

    _, _, _, _, _, predictions, scores = blink_args.main_dense.run(blink_args, None, *blink_args.models, test_data=data_to_link)
    predictions = predictions[0]  # batch_size=1
    entity_candidates = predictions[:blink_config['k']]
    entity_candidates_descriptions = []
    for entity_candidate in entity_candidates:
        text = blink_args.id2text[title2id[entity_candidate]]
        entity_candidates_descriptions.append(text[:blink_args.num_entity_description_characters])

    return entity_candidates, entity_candidates_descriptions


def test_rel_entity_candidates():
    # test generating entity-candidates and entity-candidates-descriptions from REL
    base_url = '/scratch365/yding4/REL/data/'
    wiki_version = 'wiki_2014'
    rel_config = {
       "base_url": base_url,
        "wiki_version": wiki_version,
        "k": 10,
        "num_entity_description_characters": 150,
    }
    rel_args = argparse.Namespace(**rel_config)
    '''
    mention_detection = MentionDetection(rel_args.base_url, rel_args.wiki_version)
    wikipedia = Wikipedia(rel_args.base_url, rel_args.wiki_version)
    wikipedia_yago_freq = WikipediaYagoFreq(rel_args.base_url, rel_args.wiki_version, wikipedia)
    wikipedia_yago_freq.extract_entity_description()
    '''
    rel_args.mention_detection = mention_detection
    rel_args.wikipedia = wikipedia
    rel_args.wikipedia_yago_freq = wikipedia_yago_freq

    entity_mention = 'FIFA World Cup'
    entity_candidates, entity_candidates_desciptions = rel_entity_candidates_descriptions(entity_mention, rel_args)
    print('entity_candidates:', entity_candidates)
    print('entity_candidates_desciptions:', entity_candidates_desciptions)


def test_blink():
    import blink.main_dense as main_dense
    import torch
    torch.cuda.set_device(1)

    models_path = "/afs/crc.nd.edu/user/y/yding4/EL_project/BLINK/models/"
    blink_config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": 10,
            "biencoder_model": models_path+"biencoder_wiki_large.bin",
            "biencoder_config": models_path+"biencoder_wiki_large.json",
            "entity_catalogue": models_path+"entity.jsonl",
            "entity_encoding": models_path+"all_entities_large.t7",
            "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
            "crossencoder_config": models_path+"crossencoder_wiki_large.json",
            "fast": False,  # set this to be true if speed is a concern
            "output_path": models_path + "logs/",   # logging directory
            "k": 10,
            "num_entity_description_characters": 150,
    }

    blink_args = argparse.Namespace(**blink_config)

    '''
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
    '''

    blink_args.main_dense = main_dense
    blink_args.title2id = title2id
    blink_args.id2text = id2text
    blink_args.models = models

    mention = 'FIFA World Cup'
    left_context = "England won the "
    right_context = " in 1966."
    entity_candidates, entity_candidates_descriptions = blink_entity_candidates_descriptions(mention, left_context, right_context, blink_args)
    print('entity_candidates:', entity_candidates)
    print('entity_candidates_desciptions:', entity_candidates_desciptions)
