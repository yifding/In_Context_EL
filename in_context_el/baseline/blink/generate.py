

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


