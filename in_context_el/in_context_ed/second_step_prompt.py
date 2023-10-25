import os
import json
import argparse
from tqdm import tqdm
import openai
from in_context_el.openai_key import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

import random
from in_context_el.openai_function import openai_chatgpt, openai_completion


def main():

    openai_model = 'gpt-3.5-turbo'
    openai_mode = 'chatgpt'
    if openai_mode == 'chatgpt':
        openai_function = openai_chatgpt
    elif openai_mode == 'gpt':
        openai_function = openai_completion
    else:
        raise ValueError('Unknown gpt mode')

    input_file = '/nfs/yding4/In_Context_EL/RUN_FILES/3_28_2023/multi_choice_prompt/KORE50.json'
    output_file = '/nfs/yding4/In_Context_EL/RUN_FILES/3_28_2023/multi_choice_prompt/KORE50_out.json'
    num_entity_description_characters = 150
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

        multi_choice_prompts = []
        multi_choice_prompt_results = []

        for (
            entity_mention, 
            prompt_result,
            entity_candidates,
            entity_candidates_description,
        ) in zip(
            entities['entity_mentions'], 
            entities['prompt_results'],
            entities['entity_candidates_list'],
            entities['entity_candidates_description_list'],
        ):
            
            multi_choice_prompt = ''
            for index, (entity_candidate, entity_candidate_description) in enumerate(zip(entity_candidates, entity_candidates_description)):
                description = entity_candidate + ' ' + entity_candidate_description[:num_entity_description_characters]
                multi_choice_prompt += f'({index + 1}). ' + description + '\n'
        
            multi_choice_prompt = prompt_result + '\n\n' + f'Which of the following does {entity_mention} refer to?' + '\n\n' + multi_choice_prompt
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