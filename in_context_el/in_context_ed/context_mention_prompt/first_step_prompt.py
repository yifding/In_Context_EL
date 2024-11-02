import os
import json
import argparse
from tqdm import tqdm
import openai
from in_context_el.openai_key import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
from in_context_el.openai_function import openai_chatgpt, openai_completion


def parse_args():
    parser = argparse.ArgumentParser(
        description='1st step to collect prompt for entity information.',
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
        default='/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/mention_prompt',
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
        choices=['gpt-3.5-turbo', 'text-curie-001', 'text-davinci-003', 'gpt-4', 'gpt-4o'],
        type=str,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    assert os.path.isfile(args.input_file)
    return args


def main():
    args = parse_args()
    input_file = args.input_file
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)
    num_context_characters = args.num_context_characters
    output_file = args.output_file
    openai_mode = args.openai_mode
    openai_model = args.openai_model
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