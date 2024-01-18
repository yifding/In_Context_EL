import os
import json
import argparse
import jsonlines
from tqdm import tqdm

import openai
from in_context_el.openai_key import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
from in_context_el.openai_function import openai_chatgpt, openai_completion


def parse_args():
    parser = argparse.ArgumentParser(
        description='single step to collect prompt for entity information.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        # required=True,
        default= '/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/rel_blink_candidates/aida_testb.json',
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default='/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/new_one_step_prompt',
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="output file",
        # required=True,
        default="aida_testb.json",
        type=str,
    )
    # hyper parameters:
    parser.add_argument(
        "--num_entity_description_characters",
        help="maximum number of characters of entity description",
        # required=True,
        default=150,
        type=int,
    )
    parser.add_argument(
        "--openai_mode",
        help="",
        # required=True,
        default='chatgpt',
        choices=['chatgpt', 'gpt'],
        type=str,
    )
    parser.add_argument(
        "--openai_model",
        help="",
        # required=True,
        default='ft:gpt-3.5-turbo-0613:amrit::8VNXmmdS',
        choices=[
            'gpt-3.5-turbo',
            'text-curie-001',
            'text-davinci-003',
            'gpt-4',
            'ft:gpt-3.5-turbo-0613:amrit::8VNXmmdS',
            'ft:gpt-3.5-turbo-0613:amrit:finetune-control:8fw9eH5q',
        ],
        type=str,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    assert os.path.isfile(args.input_file)
    return args


def main():
    args = parse_args()
    num_entity_description_characters = args.num_entity_description_characters
    dummy_entity = 'None of the entity match.'
    openai_model = args.openai_model
    openai_mode = args.openai_mode
    if openai_mode == 'chatgpt':
        openai_function = openai_chatgpt
    elif openai_mode == 'gpt':
        openai_function = openai_completion
    else:
        raise ValueError('Unknown gpt mode')

    input_file = args.input_file
    output_file = args.output_file
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)
        for doc_name, instance in tqdm(doc_name2instance.items()):
            sentence = instance['sentence']
            entities = instance['entities']
            multi_choice_prompts = []
            multi_choice_prompt_results = []
            for (
                start,
                end,
                entity_mention,
                entity_name,
                entity_candidates,
                entity_candidates_descriptions
            ) in zip(
                entities['starts'],
                entities['ends'],
                entities['entity_mentions'],
                entities['entity_names'],
                entities['entity_candidates'],
                entities['entity_candidates_descriptions'],
            ):
                new_sentence = sentence[max(0, start - num_entity_description_characters): end + num_entity_description_characters]
                entity_candidates = entity_candidates[:9]
                entity_candidates_descriptions = entity_candidates_descriptions[:9]
                entity_candidates.append(dummy_entity)
                entity_candidates_descriptions.append('')
                multi_choice_prompt = ''
                for index, (entity_candidate, entity_candidate_description) in enumerate(
                        zip(entity_candidates, entity_candidates_descriptions)):
                    description = entity_candidate + ' ' + entity_candidate_description[:num_entity_description_characters]
                    multi_choice_prompt += f'({index + 1}). ' + description + ' \n'

                if entity_name in entity_candidates:
                    gt_index = entity_candidates.index(entity_name)
                    gt_entity = entity_name
                else:
                    gt_index = entity_candidates.index(dummy_entity)
                    gt_entity = dummy_entity

                tmp_instance = {
                    'messages': [
                        {
                            'role': 'system',
                            'content': new_sentence
                                + ' ' + f'Which of the following entities is {entity_mention} in this sentence?'
                                + ' ' + multi_choice_prompt
                        },
                        {
                            'role': 'assistant',
                            'content': f'({gt_index + 1}). ' + gt_entity
                        },
                    ]
                }
                multi_choice_prompts.append(tmp_instance['messages'][0]['content'])
                tmp_n = 0
                while tmp_n <= 10:
                    try:
                        complete_output = openai_function(multi_choice_prompts[-1], model=openai_model)
                        break
                    except:
                        tmp_n += 1
                multi_choice_prompt_results.append(complete_output)
                entities['multi_choice_prompts'] = multi_choice_prompts
                entities['multi_choice_prompt_results'] = multi_choice_prompt_results
                doc_name2instance[doc_name]['entities'] = entities

        with open(output_file, 'w') as writer:
            json.dump(doc_name2instance, writer, indent=4)


if __name__ == '__main__':
    main()
