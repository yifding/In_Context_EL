import os
import json
import argparse
import jsonlines
import replicate
from tqdm import tqdm

from in_context_el.openai_key import REPLICATE_API_KEY

os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_KEY


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
        default='/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/new_one_step_prompt_llama2',
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
        "--replicate_model",
        help="",
        # required=True,
        default="yifding/aida_discriminative:24fdd2cb1085b68ddbac8be95a2b9e354718f5a793b80cd88e96421f5b4a5bba",
        choices=[
            "yifding/aida_discriminative:24fdd2cb1085b68ddbac8be95a2b9e354718f5a793b80cd88e96421f5b4a5bba",
            "mistralai/mixtral-8x7b-instruct-v0.1:cf18decbf51c27fed6bbdc3492312c1c903222a56e3fe9ca02d6cbe5198afc10",
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
                        output = replicate.run(
                            args.replicate_model,
                            input={'prompt': multi_choice_prompts[-1]},
                        )
                        complete_output = ''.join(output)
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
