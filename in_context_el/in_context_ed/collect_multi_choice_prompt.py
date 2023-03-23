import os
import json
import argparse
from tqdm import tqdm
import openai
from in_context_el.openai_key import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

import random
from REL.wikipedia import Wikipedia
from REL.wikipedia_yago_freq import WikipediaYagoFreq
from REL.mention_detection import MentionDetection

from in_context_el.openai_function import openai_chatgpt

def parse_args():
    parser = argparse.ArgumentParser(
        description='2nd step to collect multi-choice prompt for entity disambiguation.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/mention_prompt/KORE50.json",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/multi_choice_prompt",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="output file",
        # required=True,
        default="KORE50.json",
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
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)

    for doc_name, instance in tqdm(doc_name2instance.items()):
        entities = instance['entities']
        entity_mentions = entities['entity_mentions']
        entity_names = entities['entity_names']
        prompt_results = entities['prompt_results']

        entity_candidates = []
        multi_choice_prompts = []
        multi_choice_prompt_results = []

        for (
            entity_mention, 
            prompt_result,
        ) in zip(
            entity_mentions, 
            prompt_results,
        ):
            processed_mention = mention_detection.preprocess_mention(entity_mention)
            candidate_entities = mention_detection.get_candidates(processed_mention)
            processed_candidate_entities = [wikipedia.preprocess_ent_name(candidate_entity[0]) for candidate_entity in candidate_entities[:args.num_entity_candidates]]
            multi_choice_prompt = ''
            for index, processed_candidate_entity in enumerate(processed_candidate_entities):
                id = wikipedia.ent_wiki_id_from_name(processed_candidate_entity)
                if id in wikipedia_yago_freq.entity_id2description:
                    description = wikipedia_yago_freq.entity_id2description[id][:args.num_entity_description_characters]
                else:
                    description = processed_candidate_entity
                multi_choice_prompt += f'({index + 1}). ' + description + '\n'
        
            multi_choice_prompt = prompt_result + '\n\n' + f'Which of the following entity best describe {entity_mention} ?' + '\n\n' + multi_choice_prompt
            complete_output = openai_chatgpt(multi_choice_prompt)

            multi_choice_prompts.append(multi_choice_prompt)
            multi_choice_prompt_results.append(complete_output)
            entity_candidates.append(processed_candidate_entities)

        entities['multi_choice_prompts'] = multi_choice_prompts
        entities['multi_choice_prompt_results'] = multi_choice_prompt_results
        entities['entity_candidates'] = entity_candidates
        doc_name2instance[doc_name]['entities'] = entities

    output_file = args.output_file
    with open(output_file, 'w') as writer:
        json.dump(doc_name2instance, writer, indent=4)


if __name__ == '__main__':
    main()