import os
import json
import argparse
from tqdm import tqdm

from in_context_el.baseline.rel.generate import prepare_rel_args,rel_entity_candidates_descriptions
from in_context_el.baseline.blink.generate import prepare_blink_args, blink_entity_candidates_descriptions
from in_context_el.openai_function import openai_chatgpt

from in_context_el.in_context_ed.evaluation_raw import process_multi_choice_prompt


def entgpt_p(
    mention,
    left_context,
    right_context,
    rel_args,
    blink_args,
    num_entity_candidates=10,
    num_entity_description_characters=150,
    openai_model='gpt-3.5-turbo',
):

    rel_candidate_entities, rel_candidate_entities_descriptions = rel_entity_candidates_descriptions(
        mention,
        rel_args,
    )

    blink_candidate_entities, blink_candidate_entities_descriptions = blink_entity_candidates_descriptions(
        mention,
        left_context,
        right_context,
        blink_args,
    )


    # re-implement the logic of
    # https://github.com/yifding/In_Context_EL/blob/main/in_context_el/in_context_ed/prepare_entity_candidates/rel_blink_entity_candidates.py#L112
    if len(rel_candidate_entities) == 0:
        for blink_entity_candidate in blink_candidate_entities:
            blink_processed_entity = rel_args.wikipedia.preprocess_ent_name(blink_entity_candidate)
            id = rel_args.wikipedia.ent_wiki_id_from_name(blink_processed_entity)
            if id > 0 and blink_processed_entity not in rel_candidate_entities:
                rel_candidate_entities.append(blink_processed_entity)
                if id in rel_args.wikipedia_yago_freq.entity_id2description:
                    description = rel_args.wikipedia_yago_freq.entity_id2description[id][:num_entity_description_characters]
                else:
                    description = blink_processed_entity
                rel_candidate_entities_descriptions.append(description)

    entity_candidates = rel_candidate_entities[:num_entity_candidates]
    entity_candidates_descriptions = rel_candidate_entities_descriptions[:num_entity_candidates]

    context = left_context + mention + right_context
    prompt = context + " \n What does " + mention + " in this sentence referring to?"
    prompt_result = openai_chatgpt(prompt, model=openai_model)

    multi_choice_prompt = ''
    for index, (entity_candidate, entity_candidate_description) in enumerate(
            zip(entity_candidates, entity_candidates_descriptions)):
        description = entity_candidate + ' ' + entity_candidate_description[:num_entity_description_characters]
        multi_choice_prompt += f'({index + 1}). ' + description + '\n'

    multi_choice_prompt = prompt_result + '\n\n' + f'Which of the following entities is {mention} in this sentence?' + '\n\n' + multi_choice_prompt
    multi_choice_prompt_result = openai_chatgpt(multi_choice_prompt, model=openai_model)

    predict_entity_name = process_multi_choice_prompt(multi_choice_prompt_result, entity_candidates)

    return {
        'predict_entity_name': predict_entity_name,
        'entity_candidates': entity_candidates,
        'entity_candidates_descriptions': entity_candidates_descriptions,
        'prompt': prompt,
        'prompt_result': prompt_result,
        'multi_choice_prompt': multi_choice_prompt,
        'multi_choice_prompt_result': multi_choice_prompt_result,
    }


def entgpt_i(
    mention,
    left_context,
    right_context,
    rel_args,
    blink_args,
    num_entity_candidates=10,
    num_entity_description_characters=150,
    openai_model='ft:gpt-3.5-turbo-0613:amrit::8VNXmmdS',
):
    rel_candidate_entities, rel_candidate_entities_descriptions = rel_entity_candidates_descriptions(
        mention,
        rel_args,
    )

    blink_candidate_entities, blink_candidate_entities_descriptions = blink_entity_candidates_descriptions(
        mention,
        left_context,
        right_context,
        blink_args,
    )

    # re-implement the logic of
    # https://github.com/yifding/In_Context_EL/blob/main/in_context_el/in_context_ed/prepare_entity_candidates/rel_blink_entity_candidates.py#L112
    if len(rel_candidate_entities) == 0:
        for blink_entity_candidate in blink_candidate_entities:
            blink_processed_entity = rel_args.wikipedia.preprocess_ent_name(blink_entity_candidate)
            id = rel_args.wikipedia.ent_wiki_id_from_name(blink_processed_entity)
            if id > 0 and blink_processed_entity not in rel_candidate_entities:
                rel_candidate_entities.append(blink_processed_entity)
                if id in rel_args.wikipedia_yago_freq.entity_id2description:
                    description = rel_args.wikipedia_yago_freq.entity_id2description[id][:num_entity_description_characters]
                else:
                    description = blink_processed_entity
                rel_candidate_entities_descriptions.append(description)

    entity_candidates = rel_candidate_entities[:num_entity_candidates-1]
    entity_candidates_descriptions = rel_candidate_entities_descriptions[:num_entity_candidates-1]
    dummy_entity = 'None of the entity match.'
    entity_candidates.append(dummy_entity)
    entity_candidates_descriptions.append('')

    multi_choice_prompt = ''
    for index, (entity_candidate, entity_candidate_description) in enumerate(
            zip(entity_candidates, entity_candidates_descriptions)):
        description = entity_candidate + ' ' + entity_candidate_description[:num_entity_description_characters]
        multi_choice_prompt += f'({index + 1}). ' + description + ' \n'

    context = left_context + mention + right_context
    prompt = context + ' ' + f'Which of the following entities is {mention} in this sentence?' + ' ' + multi_choice_prompt
    prompt_result = openai_chatgpt(prompt, model=openai_model)
    predict_entity_name = process_multi_choice_prompt(prompt_result, entity_candidates)

    return {
        'predict_entity_name': predict_entity_name,
        'entity_candidates': entity_candidates,
        'entity_candidates_descriptions': entity_candidates_descriptions,
        'multi_choice_prompt': prompt,
        'multi_choice_prompt_result': prompt_result,
    }


def test_entgpt_p():
    # '''
    rel_args = prepare_rel_args()
    blink_args = prepare_blink_args()
    # '''

    num_context_characters = 150
    num_entity_description_characters = 150
    num_entity_candidates = 10
    openai_model='gpt-3.5-turbo'

    datasets = ['ace2004', 'aquaint', 'msnbc', 'aida_test', 'clueweb', 'wikipedia']
    input_dir = '/scratch365/yding4/In_Context_EL/RUN_FILES/11_14_2024/ED_standard_datasets'
    output_dir = '/scratch365/yding4/In_Context_EL/RUN_FILES/11_14_2024/baselines/EntGPT-P/ED_standard_datasets/predictions'
    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        print('dataset:', dataset)
        input_file = os.path.join(input_dir, dataset + '.json')
        output_file = os.path.join(output_dir, dataset + '.json')
        with open(input_file) as reader:
            doc_name2instance = json.load(reader)

        if os.path.isfile(output_file):
            with open(output_file) as reader:
                doc_name2instance = json.load(reader)

        for doc_name, instance in tqdm(doc_name2instance.items()):
            if 'out_dict' in instance:
                continue

            sentence = instance['sentence']
            entities = instance['entities']
            stats = entities['starts']
            ends = entities['ends']

            predict_entity_names = []
            entity_candidates_list = []
            entity_candidates_descriptions_list = []
            prompts = []
            prompt_results = []
            multi_choice_prompts = []
            multi_choice_prompt_results = []

            for start, end in zip(stats, ends):
                mention = sentence[start: end]
                left_context = sentence[max(0, start - num_context_characters): start]
                right_context = sentence[end: end + num_context_characters]

                out_dict = entgpt_p(
                    mention,
                    left_context,
                    right_context,
                    rel_args,
                    blink_args,
                    num_entity_candidates=num_entity_candidates,
                    num_entity_description_characters=num_entity_description_characters,
                    openai_model=openai_model,
                )

                predict_entity_names.append(out_dict['predict_entity_name'])
                prompts.append(out_dict['prompt'])
                prompt_results.append(out_dict['prompt_result'])
                multi_choice_prompts.append(out_dict['multi_choice_prompt'])
                multi_choice_prompt_results.append(out_dict['multi_choice_prompt_result'])
                entity_candidates_list.append(out_dict['entity_candidates'])
                entity_candidates_descriptions_list.append(out_dict['entity_candidates_descriptions'])


            # doc_name2instance[doc_name]['entities'] = entities
            doc_name2instance[doc_name]['pred_entities'] = {
                'starts': entities['starts'],
                'ends': entities['ends'],
                'entity_mentions': entities['entity_mentions'],
                'entity_names': predict_entity_names,
            }

            doc_name2instance[doc_name]['out_dict'] = {
                'entity_candidates': entity_candidates_list,
                'entity_candidates_descriptions': entity_candidates_descriptions_list,
                'prompts': prompts,
                'prompt_results': prompt_results,
                'multi_choice_prompts': multi_choice_prompts,
                'multi_choice_prompt_results': multi_choice_prompt_results,
            }

            with open(output_file, 'w') as writer:
                json.dump(doc_name2instance, writer, indent=4)


def test_entgpt_i():
    # '''
    rel_args = prepare_rel_args()
    blink_args = prepare_blink_args()
    # '''

    num_context_characters = 150
    num_entity_description_characters = 150
    num_entity_candidates = 10
    openai_model='ft:gpt-3.5-turbo-0613:amrit::8VNXmmdS'

    # datasets = ['ace2004', 'aquaint', 'msnbc', 'aida_test', 'clueweb', 'wikipedia']
    datasets = ['ace2004']
    input_dir = '/scratch365/yding4/In_Context_EL/RUN_FILES/11_14_2024/ED_standard_datasets'
    output_dir = '/scratch365/yding4/In_Context_EL/RUN_FILES/11_14_2024/baselines/EntGPT-I/ED_standard_datasets/predictions'
    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        print('dataset:', dataset)
        input_file = os.path.join(input_dir, dataset + '.json')
        output_file = os.path.join(output_dir, dataset + '.json')
        with open(input_file) as reader:
            doc_name2instance = json.load(reader)

        if os.path.isfile(output_file):
            with open(output_file) as reader:
                doc_name2instance = json.load(reader)

        for doc_name, instance in tqdm(doc_name2instance.items()):
            if 'out_dict' in instance:
                continue

            sentence = instance['sentence']
            entities = instance['entities']
            stats = entities['starts']
            ends = entities['ends']

            predict_entity_names = []
            entity_candidates_list = []
            entity_candidates_descriptions_list = []
            prompts = []
            prompt_results = []
            multi_choice_prompts = []
            multi_choice_prompt_results = []

            for start, end in zip(stats, ends):
                mention = sentence[start: end]
                left_context = sentence[max(0, start - num_context_characters): start]
                right_context = sentence[end: end + num_context_characters]

                out_dict = entgpt_i(
                    mention,
                    left_context,
                    right_context,
                    rel_args,
                    blink_args,
                    num_entity_candidates=num_entity_candidates,
                    num_entity_description_characters=num_entity_description_characters,
                    openai_model=openai_model,
                )

                predict_entity_names.append(out_dict['predict_entity_name'])
                # prompts.append(out_dict['prompt'])
                # prompt_results.append(out_dict['prompt_result'])
                multi_choice_prompts.append(out_dict['multi_choice_prompt'])
                multi_choice_prompt_results.append(out_dict['multi_choice_prompt_result'])
                entity_candidates_list.append(out_dict['entity_candidates'])
                entity_candidates_descriptions_list.append(out_dict['entity_candidates_descriptions'])


            # doc_name2instance[doc_name]['entities'] = entities
            doc_name2instance[doc_name]['pred_entities'] = {
                'starts': entities['starts'],
                'ends': entities['ends'],
                'entity_mentions': entities['entity_mentions'],
                'entity_names': predict_entity_names,
            }

            doc_name2instance[doc_name]['out_dict'] = {
                'entity_candidates': entity_candidates_list,
                'entity_candidates_descriptions': entity_candidates_descriptions_list,
                # 'prompts': prompts,
                # 'prompt_results': prompt_results,
                'multi_choice_prompts': multi_choice_prompts,
                'multi_choice_prompt_results': multi_choice_prompt_results,
            }

            with open(output_file, 'w') as writer:
                json.dump(doc_name2instance, writer, indent=4)



# if __name__ == '__main__':
    # test_entgpt_i()

'''
rel_args = prepare_rel_args()
blink_args = prepare_blink_args()
'''

num_context_characters = 150
num_entity_description_characters = 150
num_entity_candidates = 10
openai_model = 'ft:gpt-3.5-turbo-0613:amrit::8VNXmmdS'

# datasets = ['ace2004', 'aquaint', 'msnbc', 'aida_test', 'clueweb', 'wikipedia']
datasets = ['ace2004']
input_dir = '/scratch365/yding4/In_Context_EL/RUN_FILES/11_14_2024/ED_standard_datasets'
output_dir = '/scratch365/yding4/In_Context_EL/RUN_FILES/11_14_2024/baselines/EntGPT-I/ED_standard_datasets/predictions'
os.makedirs(output_dir, exist_ok=True)

for dataset in datasets:
    print('dataset:', dataset)
    input_file = os.path.join(input_dir, dataset + '.json')
    output_file = os.path.join(output_dir, dataset + '.json')
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)

    if os.path.isfile(output_file):
        with open(output_file) as reader:
            doc_name2instance = json.load(reader)

    for doc_name, instance in tqdm(doc_name2instance.items()):
        if 'out_dict' in instance:
            continue

        sentence = instance['sentence']
        entities = instance['entities']
        stats = entities['starts']
        ends = entities['ends']

        predict_entity_names = []
        entity_candidates_list = []
        entity_candidates_descriptions_list = []
        prompts = []
        prompt_results = []
        multi_choice_prompts = []
        multi_choice_prompt_results = []

        for start, end in zip(stats, ends):
            mention = sentence[start: end]
            left_context = sentence[max(0, start - num_context_characters): start]
            right_context = sentence[end: end + num_context_characters]

            out_dict = entgpt_i(
                mention,
                left_context,
                right_context,
                rel_args,
                blink_args,
                num_entity_candidates=num_entity_candidates,
                num_entity_description_characters=num_entity_description_characters,
                openai_model=openai_model,
            )

            predict_entity_names.append(out_dict['predict_entity_name'])
            # prompts.append(out_dict['prompt'])
            # prompt_results.append(out_dict['prompt_result'])
            multi_choice_prompts.append(out_dict['multi_choice_prompt'])
            multi_choice_prompt_results.append(out_dict['multi_choice_prompt_result'])
            entity_candidates_list.append(out_dict['entity_candidates'])
            entity_candidates_descriptions_list.append(out_dict['entity_candidates_descriptions'])

        # doc_name2instance[doc_name]['entities'] = entities
        doc_name2instance[doc_name]['pred_entities'] = {
            'starts': entities['starts'],
            'ends': entities['ends'],
            'entity_mentions': entities['entity_mentions'],
            'entity_names': predict_entity_names,
        }

        doc_name2instance[doc_name]['out_dict'] = {
            'entity_candidates': entity_candidates_list,
            'entity_candidates_descriptions': entity_candidates_descriptions_list,
            # 'prompts': prompts,
            # 'prompt_results': prompt_results,
            'multi_choice_prompts': multi_choice_prompts,
            'multi_choice_prompt_results': multi_choice_prompt_results,
        }

        with open(output_file, 'w') as writer:
            json.dump(doc_name2instance, writer, indent=4)