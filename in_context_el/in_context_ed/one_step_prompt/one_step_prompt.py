import os
import json
import argparse
from tqdm import tqdm
import openai
from in_context_el.openai_key import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# import random
from in_context_el.openai_function import openai_chatgpt, openai_completion

def parse_args():
    parser = argparse.ArgumentParser(
        description='one step to collect prompt for entity information.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        # required=True,
        default= '/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/rel_blink_candidates/ace2004.json',
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default='/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/one_step_prompt',
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="output file",
        # required=True,
        default="ace2004.json",
        type=str,
    )
    # hyper parameters:
    parser.add_argument(
        "--num_context_characters",
        help="maximum number of characters of original input sentence around mention",
        # required=True,
        default=150,
        type=int,
    )
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
        default='gpt-3.5-turbo',
        choices=['gpt-3.5-turbo', 'text-curie-001', 'text-davinci-003'],
        type=str,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    assert os.path.isfile(args.input_file)
    return args


def main():
    args = parse_args()
    openai_model = args.openai_model
    openai_mode = args.openai_mode
    num_context_characters = args.num_context_characters
    if openai_mode == 'chatgpt':
        openai_function = openai_chatgpt
    elif openai_mode == 'gpt':
        openai_function = openai_completion
    else:
        raise ValueError('Unknown gpt mode')

    input_file = args.input_file
    output_file = args.output_file
    num_entity_description_characters = args.num_entity_description_characters
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)

    # consider continue querying when bug occurs
    if os.path.isfile(output_file):
        with open(output_file) as reader:
            exist_doc_name2instance = json.load(reader)
        exist_doc_names = list(exist_doc_name2instance.keys())
    else:
        exist_doc_names = []

    for doc_name, instance in tqdm(doc_name2instance.items()):
        if doc_name in exist_doc_names and 'multi_choice_prompts' in exist_doc_name2instance[doc_name]['entities']:
            doc_name2instance[doc_name]['entities'] = exist_doc_name2instance[doc_name]['entities']
            continue

        entities = instance['entities']
        sentence = instance['sentence']

        multi_choice_prompts = []
        multi_choice_prompt_results = []

        for (
            start,
            end,
            entity_mention, 
            prompt_result,
            entity_candidates,
            entity_candidates_description,
        ) in zip(
            entities['starts'],
            entities['ends'],
            entities['entity_mentions'], 
            entities['prompt_results'],
            entities['entity_candidates'],
            entities['entity_candidates_descriptions'],
        ):
            
            multi_choice_prompt = ''
            for index, (entity_candidate, entity_candidate_description) in enumerate(zip(entity_candidates, entity_candidates_description)):
                description = entity_candidate + ' ' + entity_candidate_description[:num_entity_description_characters]
                multi_choice_prompt += f'({index + 1}). ' + description + '\n'
        
            # multi_choice_prompt = prompt_result + '\n\n' + f'Which of the following entities does {entity_mention} refer to in this sentence?' + '\n\n' + multi_choice_prompt
            # multi_choice_prompt = prompt_result + '\n\n' + f'Which of the following entity best describe {entity_mention} ?' + '\n\n' + multi_choice_prompt
            
            
            # multi_choice_prompt = prompt_result + '\n\n' + f'Which of the following entities is {entity_mention} in this sentence?' + '\n\n' + multi_choice_prompt
            
            prompt_sentence = sentence[max(0, start - num_context_characters): start] + entity_mention + sentence[end: end + num_context_characters]
            multi_choice_prompt = prompt_sentence + '\n\n' + f'Which of the following entities is {entity_mention} in this sentence?' + '\n\n' + multi_choice_prompt


            complete_output = openai_function(multi_choice_prompt, model=openai_model)

            multi_choice_prompts.append(multi_choice_prompt)
            multi_choice_prompt_results.append(complete_output)

        entities['multi_choice_prompts'] = multi_choice_prompts
        entities['multi_choice_prompt_results'] = multi_choice_prompt_results
        doc_name2instance[doc_name]['entities'] = entities

    
        with open(output_file, 'w') as writer:
            json.dump(doc_name2instance, writer, indent=4)


if __name__ == '__main__':
    main()