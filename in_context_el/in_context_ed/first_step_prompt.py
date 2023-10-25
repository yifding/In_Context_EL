import os
import json
from tqdm import tqdm
import openai
from in_context_el.openai_key import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
from in_context_el.openai_function import openai_chatgpt, openai_completion

def main():
    input_file = '/nfs/yding4/In_Context_EL/RUN_FILES/3_28_2023/mention_prompt/KORE50.json'
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)
    output_file = '/nfs/yding4/In_Context_EL/RUN_FILES/3_28_2023/multi_choice_prompt/KORE50.json'
    num_context_characters = 150
    openai_mode = 'chatgpt'
    openai_model = 'gpt-3.5-turbo'
    if openai_mode == 'chatgpt':
        openai_function = openai_chatgpt
    elif openai_mode == 'gpt':
        openai_function = openai_completion
    else:
        raise ValueError('Unknown gpt mode')

    # consider continue querying when bug occurs
    if os.path.isfile(output_file):
        with open(output_file) as reader:
            exist_doc_name2instance = json.load(reader)
        exist_doc_names = list(exist_doc_name2instance.keys())
    else:
        exist_doc_names = []

    for doc_name, instance in tqdm(doc_name2instance.items()):
        if doc_name in exist_doc_names and 'prompt_results' in exist_doc_name2instance[doc_name]['entities']:
            doc_name2instance[doc_name]['entities'] = exist_doc_name2instance[doc_name]['entities']
            continue
        entities = instance['entities']
        entity_mentions = entities['entity_mentions']
        starts = entities['starts']
        ends = entities['ends']
        sentence = instance['sentence']
        prompt_results = []
        prompts = []
        for (
            entity_mention,
            start,
            end
        ) in zip(
            entity_mentions,
            starts,
            ends,
        ):
            prompt_sentence = sentence[max(0, start - num_context_characters): start] + entity_mention + sentence[end: end + num_context_characters]
            prompt = prompt_sentence + " \n What does " + entity_mention + " in this sentence referring to?"
            prompts.append(prompt)
            complete_output = openai_function(prompt, model=openai_model)
            prompt_results.append(complete_output)
            
        entities['prompts'] = prompts
        entities['prompt_results'] = prompt_results
        doc_name2instance[doc_name]['entities'] = entities

        with open(output_file, 'w') as writer:
            json.dump(doc_name2instance, writer, indent=4)


if __name__ == '__main__':
    main()