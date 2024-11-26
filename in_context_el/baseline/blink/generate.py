import os
import argparse
import blink.main_dense as main_dense


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
    entity_candidates = predictions[:blink_args.k]
    entity_candidates_descriptions = []
    for entity_candidate in entity_candidates:
        text = blink_args.id2text[blink_args.title2id[entity_candidate]]
        entity_candidates_descriptions.append(text[:blink_args.num_entity_description_characters])

    return entity_candidates, entity_candidates_descriptions


def prepare_blink_args(
    models_path="/afs/crc.nd.edu/user/y/yding4/EL_project/BLINK/models/",
    k=10,
    gpu_device=1,
    num_entity_description_characters=150,
):
    import blink.main_dense as main_dense
    import torch
    if gpu_device >= 0:
        torch.cuda.set_device(gpu_device)

    blink_config = {
        "test_entities": None,
        "test_mentions": None,
        "interactive": False,
        "top_k": k,
        "biencoder_model": models_path + "biencoder_wiki_large.bin",
        "biencoder_config": models_path + "biencoder_wiki_large.json",
        "entity_catalogue": models_path + "entity.jsonl",
        "entity_encoding": models_path + "all_entities_large.t7",
        "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
        "crossencoder_config": models_path + "crossencoder_wiki_large.json",
        "fast": False,  # set this to be true if speed is a concern
        "output_path": models_path + "logs/",  # logging directory
        "k": k,
        "num_entity_description_characters": num_entity_description_characters,
    }

    blink_args = argparse.Namespace(**blink_config)
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

    blink_args.main_dense = main_dense
    blink_args.title2id = title2id
    blink_args.id2text = id2text
    blink_args.models = models

    return blink_args


def test_blink():
    # blink_args = prepare_blink_args()
    mention = 'FIFA World Cup'
    left_context = "England won the "
    right_context = " in 1966."
    entity_candidates, entity_candidates_descriptions = blink_entity_candidates_descriptions(
        mention,
        left_context,
        right_context,
        blink_args,
    )

    print('entity_candidates:', entity_candidates)
    print('entity_candidates_descriptions:', entity_candidates_descriptions)


def blink4ed(
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
    ):

    starts = []
    ends = []
    entity_mentions = []
    entity_names = []
    entity_candidates = []
    entity_candidates_descriptions = []

    raw_predictions = []
    raw_scores = []
    for span in spans:
        assert len(span) == 2
        start, end = span
        mention = sentence[start: end]
        left_context = sentence[max(0, start-num_context_characters): start]
        right_context = sentence[end: end+num_context_characters]


        data_to_link = [{
            "id": 0,
            "label": "unknown",
            "label_id": -1,
            "context_left": left_context,
            "mention": mention,
            "context_right": right_context,
        }]

        _, _, _, _, _, predictions, scores = main_dense.run(blink_args, None, *models, test_data=data_to_link)
        predictions = predictions[0] # batch_size=1
        raw_predictions.append(predictions)
        raw_scores.append(scores)

        starts.append(start)
        ends.append(end)
        entity_mentions.append(mention)
        entity_names.append(predictions[0])
        tmp_entity_candidates = predictions[:k]
        entity_candidates.append(tmp_entity_candidates)

        tmp_entity_candidates_descriptions = []
        for entity_candidate in tmp_entity_candidates:
            text = id2text[title2id[entity_candidate]]
            tmp_entity_candidates_descriptions.append(text[:num_entity_description_characters])
        entity_candidates_descriptions.append(tmp_entity_candidates_descriptions)


    if return_ori:
        return raw_predictions, raw_scores

    pred_entities = {
        'starts': starts,
        'ends': ends,
        'entity_mentions': entity_mentions,
        'entity_names': entity_names,
        'entity_candidates': entity_candidates,
        'entity_candidates_descriptions': entity_candidates_descriptions,
    }

    return pred_entities


if __name__ == '__main__':
    test_blink()

