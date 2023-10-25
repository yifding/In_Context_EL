import os
import json
import argparse
from tqdm import tqdm

import random
from REL.wikipedia import Wikipedia
from REL.wikipedia_yago_freq import WikipediaYagoFreq
from REL.mention_detection import MentionDetection


def parse_args():
    parser = argparse.ArgumentParser(
        description='2nd step to collect entity candidates for entity disambiguation by combining blink and rel entity candidates.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/blink_candidates/ace2004.json",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/rel_blink_candidates",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="output file",
        # required=True,
        default="ace2004.json",
        type=str,
    )
    parser.add_argument(
        "--base_url",
        help="input directory for wikipedia used of REL",
        # required=True,
        default="/nfs/yding4/REL/data/",
        type=str,
    )
    parser.add_argument(
        "--wiki_version",
        help="specific version for wikipedia used of REL",
        # required=True,
        default="wiki_2014",
        type=str,
    )
    parser.add_argument(
        "--num_entity_candidates",
        help="maximum number of entity candidates of each mention",
        # required=True,
        default=10,
        type=int,
    )

    parser.add_argument(
        "--num_entity_description_characters",
        help="maximum number of characters allowed for each entity candidate",
        # required=True,
        default=150,
        type=int,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    assert os.path.isfile(args.input_file)
    return args


def main():
    args = parse_args()

    # module loading, take a long time to run
    mention_detection = MentionDetection(args.base_url, args.wiki_version)
    wikipedia = Wikipedia(args.base_url, args.wiki_version)
    wikipedia_yago_freq = WikipediaYagoFreq(args.base_url, args.wiki_version, wikipedia)
    wikipedia_yago_freq.extract_entity_description()

    input_file = args.input_file
    output_file = args.output_file
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)

    for doc_name, instance in tqdm(doc_name2instance.items()):
        entities = instance['entities']

        entity_candidates = []
        new_entity_names = []
        entity_candidates_descriptions = []

        for (
            entity_mention, 
            blink_entity_candidates,
            entity_name,
        ) in zip(
            entities['entity_mentions'], 
            entities['blink_entity_candidates_list'],
            entities['entity_names'],
        ):
            new_entity_name = wikipedia.preprocess_ent_name(entity_name)
            new_entity_names.append(new_entity_name)
            processed_mention = mention_detection.preprocess_mention(entity_mention)
            rel_candidate_entities = mention_detection.get_candidates(processed_mention)
            rel_processed_candidate_entities = [wikipedia.preprocess_ent_name(rel_candidate_entity[0]) for rel_candidate_entity in rel_candidate_entities[:args.num_entity_candidates]]

            # if rel entity candidates are empty, fill in valid blink entity candidates
            if len(rel_processed_candidate_entities) == 0:
                for blink_entity_candidate in blink_entity_candidates:
                    blink_processed_entity = wikipedia.preprocess_ent_name(blink_entity_candidate)
                    id = wikipedia.ent_wiki_id_from_name(blink_processed_entity)
                    if id > 0 and blink_processed_entity not in rel_processed_candidate_entities:
                        rel_processed_candidate_entities.append(blink_processed_entity)
            
            rel_processed_candidate_entities = rel_processed_candidate_entities[:args.num_entity_candidates]
            descriptions = []
            for rel_processed_candidate_entity in rel_processed_candidate_entities:
                id = wikipedia.ent_wiki_id_from_name(rel_processed_candidate_entity)
                if id in wikipedia_yago_freq.entity_id2description:
                    description = wikipedia_yago_freq.entity_id2description[id][:args.num_entity_description_characters]
                else:
                    description = rel_processed_candidate_entity
                descriptions.append(description)

            # add ground truth entity and corresponding description in the list.
            gt_entity = new_entity_name
            gt_id = wikipedia.ent_wiki_id_from_name(gt_entity)
            if gt_id in wikipedia_yago_freq.entity_id2description:
                gt_description = wikipedia_yago_freq.entity_id2description[gt_id][:args.num_entity_description_characters]
            else:
                gt_description = rel_processed_candidate_entity
            
            random_index = random.randint(0, len(rel_processed_candidate_entities) - 1)
            if gt_entity not in rel_processed_candidate_entities:
                rel_processed_candidate_entities[random_index] = gt_entity
                descriptions[random_index] = gt_description

            entity_candidates.append(rel_processed_candidate_entities)
            entity_candidates_descriptions.append(descriptions)

        entities['entity_names'] = new_entity_names
        entities['entity_candidates'] = entity_candidates
        entities['entity_candidates_descriptions'] = entity_candidates_descriptions
        doc_name2instance[doc_name]['entities'] = entities
    
        with open(output_file, 'w') as writer:
            json.dump(doc_name2instance, writer, indent=4)


if __name__ == '__main__':
    main()