import os
import argparse


def rel_entity_candidates_descriptions(mention, rel_args):
    processed_mention = rel_args.mention_detection.preprocess_mention(mention)
    rel_candidate_entities = rel_args.mention_detection.get_candidates(processed_mention)
    rel_processed_candidate_entities = [rel_args.wikipedia.preprocess_ent_name(rel_candidate_entity[0]) for rel_candidate_entity
                                        in rel_candidate_entities[:rel_args.k]]
    descriptions = []
    for rel_processed_candidate_entity in rel_processed_candidate_entities:
        id = rel_args.wikipedia.ent_wiki_id_from_name(rel_processed_candidate_entity)
        if id in rel_args.wikipedia_yago_freq.entity_id2description:
            description = rel_args.wikipedia_yago_freq.entity_id2description[id][:rel_args.num_entity_description_characters]
        else:
            description = ""
        descriptions.append(description)
    return rel_processed_candidate_entities, descriptions


def prepare_rel_args(
    base_url='/scratch365/yding4/REL/data/',    # must end with '/'
    wiki_version='wiki_2014',
    k=10,
    num_entity_description_characters=150,
):
    from REL.wikipedia import Wikipedia
    from REL.wikipedia_yago_freq import WikipediaYagoFreq
    from REL.mention_detection import MentionDetection

    rel_config = {
        "base_url": base_url,
        "wiki_version": wiki_version,
        "k": k,
        "num_entity_description_characters": num_entity_description_characters,
    }

    rel_args = argparse.Namespace(**rel_config)
    mention_detection = MentionDetection(rel_args.base_url, rel_args.wiki_version)
    wikipedia = Wikipedia(rel_args.base_url, rel_args.wiki_version)
    wikipedia_yago_freq = WikipediaYagoFreq(rel_args.base_url, rel_args.wiki_version, wikipedia)
    wikipedia_yago_freq.extract_entity_description()
    rel_args.mention_detection = mention_detection
    rel_args.wikipedia = wikipedia
    rel_args.wikipedia_yago_freq = wikipedia_yago_freq

    return rel_args


def test_rel_entity_candidates():
    rel_args = prepare_rel_args()
    entity_mention = 'FIFA World Cup'
    entity_candidates, entity_candidates_descriptions = rel_entity_candidates_descriptions(entity_mention, rel_args)
    print('entity_candidates:', entity_candidates)
    print('entity_candidates_descriptions:', entity_candidates_descriptions)


if __name__ == '__main__':
    test_rel_entity_candidates()

