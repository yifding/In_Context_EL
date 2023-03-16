import os
import json
import argparse
from tqdm import tqdm
import openai
from in_context_el.openai_key import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
from in_context_el.openai_function import openai_chatgpt
from in_context_el.dataset_reader import dataset_loader


def parse_args():
    parser = argparse.ArgumentParser(
        description='1st step to collect prompt for entity information.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        help="the extension file used by load_dataset function to load dataset",
        # required=True,
        choices=["tsv", "oke_2015", "oke_2016", "n3"],
        default="tsv",
        type=str,
    )
    parser.add_argument(
        "--key",
        help="the split key of aida-conll dataset",
        # required=True,
        choices=["", "testa", "testb"],
        default="",
        type=str,
    )
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        # required=True,
        default="/nfs/yding4/EL_project/dataset/KORE50/AIDA.tsv",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/mention_prompt",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="output file",
        # required=True,
        default="KORE50.json",
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
    # parser.add_argument(
    #     "--num_entity_candidates",
    #     help="maximum number of entity candidates of each mention",
    #     # required=True,
    #     default=10,
    #     type=int,
    # )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    assert os.path.isfile(args.input_file)
    return args


def main():
    args = parse_args()
    # load dataset
    doc_name2instance = dataset_loader(args.input_file, key=args.key, mode=args.mode)
    output_file = args.output_file
    num_context_characters = args.num_context_characters

    for doc_name, instance in tqdm(doc_name2instance.items()):
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
            complete_output = openai_chatgpt(prompt)
            prompt_results.append(complete_output)
            
        entities['prompts'] = prompts
        entities['prompt_results'] = prompt_results
        doc_name2instance[doc_name]['entities'] = entities

    with open(output_file, 'w') as writer:
        json.dump(doc_name2instance, writer, indent=4)


if __name__ == '__main__':
    main()