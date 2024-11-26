import os
import json
import argparse
from tqdm import tqdm
from copy import deepcopy

from in_context_el.openai_function import openai_chatgpt
from in_context_el.baseline.rel.generate import prepare_rel_args,rel_entity_candidates_descriptions
from in_context_el.baseline.blink.generate import prepare_blink_args, blink_entity_candidates_descriptions
from in_context_el.openai_function import openai_chatgpt
from in_context_el.in_context_ed.evaluation_raw import process_multi_choice_prompt



def llm4ed(
    mention,
    left_context,
    right_context,
    doc_name='',
    rel_args=None,
    blink_args=None,
    add_doc_name=True,
    add_indicator=True,
    add_augmented_context=True,
    add_false_option=True,
    verify_entity=True,
    max_rel_entity_candidates=6,
    max_blink_entity_candidates=3,
    augment_context_openai_model='gpt-4o',
    multi_choice_openai_model='gpt-4o',
    verify_openai_model='gpt-4o',
    num_context_characters=150,
    num_entity_description_characters=150,
):

    # obtain the context
    if add_indicator:
        context = '<sentence>' + ' ' + left_context + ' ' + '<mention>' + \
                  ' ' + mention + ' ' + '</mention>' + ' ' + right_context + ' ' + '</sentence>'
        if add_doc_name:
            context = '<doc_name>' + ' ' + doc_name + ' ' + '</doc_name>' + ' ' + context
    else:
        context = left_context + mention + right_context
        if add_doc_name:
            context = doc_name + ' ' + context

    # obtain the augmented context
    augmented_context = ''
    augmented_context_prompt = ''
    if add_augmented_context:
        if add_indicator:
            augmented_context_prompt = context + " \n What does " + mention + " (identified by <mention> </mention>)" \
                                       " in this sentence (identified by <sentence> </sentence>) referring to?"
        else:
            augmented_context_prompt = context + " \n What does " + mention + " in this sentence referring to?"
        augmented_context = openai_chatgpt(augmented_context_prompt, model=augment_context_openai_model)

    # obtain entities
    entity_candidates = []
    entity_candidates_descriptions = []

    # obtain rel entities
    if max_rel_entity_candidates > 0:
        assert rel_args is not None
        tmp_rel_candidate_entities, tmp_rel_candidate_entities_descriptions = rel_entity_candidates_descriptions(
            mention,
            rel_args,
        )

        tmp_rel_candidate_entities = tmp_rel_candidate_entities[:max_rel_entity_candidates]
        tmp_rel_candidate_entities_descriptions = tmp_rel_candidate_entities_descriptions[:max_rel_entity_candidates]

        entity_candidates.extend(tmp_rel_candidate_entities)
        entity_candidates_descriptions.extend(tmp_rel_candidate_entities_descriptions)

    # obtain blink entities
    if max_blink_entity_candidates > 0:
        assert blink_args is not None

        if mention in augmented_context:
            tmp_start = augmented_context.index(mention)
            tmp_end = start + len(mention)
            prompt_left_context = augmented_context[max(0, tmp_start - num_context_characters): tmp_start]
            prompt_right_context = augmented_context[tmp_end: tmp_end + num_context_characters]
        else:
            prompt_left_context = left_context
            prompt_right_context = right_context

    tmp_blink_entity_candidates, tmp_blink_entity_candidates_descriptions = blink_entity_candidates_descriptions(
            mention, prompt_left_context, prompt_right_context, blink_args)
    tmp_blink_entity_candidates = tmp_blink_entity_candidates[:max_blink_entity_candidates]
    tmp_blink_entity_candidates_descriptions = tmp_blink_entity_candidates_descriptions[:max_blink_entity_candidates]

    for tmp_blink_entity_candidate, tmp_blink_entity_candidates_description in zip(
        tmp_blink_entity_candidates, tmp_blink_entity_candidates_descriptions,
    ):
        if tmp_blink_entity_candidate not in entity_candidates:
            entity_candidates.append(tmp_blink_entity_candidate)
            entity_candidates_descriptions.append(tmp_blink_entity_candidates_description)

    if add_false_option:
        entity_candidates.append('None of the entity match.')
        entity_candidates_descriptions.append('')

    multi_choice_prompt = ''
    for index, (entity_candidate, entity_candidate_description) in enumerate(
            zip(entity_candidates, entity_candidates_descriptions)
    ):
        description = entity_candidate + ' ' + entity_candidate_description[:num_entity_description_characters]
        multi_choice_prompt += f'({index + 1}). ' + description + '\n'

    multi_choice_context = context
    if augmented_context_prompt:
        if add_indicator:
            multi_choice_context = multi_choice_context + ' ' + '<info>' + ' ' + augmented_context + ' ' + '</info>'
        else:
            multi_choice_context = multi_choice_context + ' ' + augmented_context

    if add_indicator:
        if augmented_context_prompt:
            multi_choice_prompt = multi_choice_context + ' ' + \
                f'Which of the following entities is {mention} (identified by <mention> </mention>) in this sentence' \
                f' (identified by <sentence> </sentence>) supplemented by the information ' \
                f'(identified by <info> </info>) ?' + ' ' + multi_choice_prompt
        else:
            multi_choice_prompt = multi_choice_context + ' ' + \
                f'Which of the following entities is {mention} (identified by <mention> </mention>) in this sentence' \
                f' (identified by <sentence> </sentence>) ?' + ' ' + multi_choice_prompt
    else:
        multi_choice_prompt = multi_choice_context + ' ' + f'Which of the following entities is {mention} ' \
            f'in this sentence?' \

    multi_choice_prompt_result = openai_chatgpt(multi_choice_prompt, model=multi_choice_openai_model)
    predict_entity_name = process_multi_choice_prompt(multi_choice_prompt_result, entity_candidates)

    out_dict = {
        'predict_entity_name': predict_entity_name,
        'entity_candidates': entity_candidates,
        'entity_candidates_descriptions': entity_candidates_descriptions,
        'prompt': augmented_context_prompt,
        'prompt_result': augmented_context,
        'multi_choice_prompt': multi_choice_prompt,
        'multi_choice_prompt_result': multi_choice_prompt_result,
    }

    if verify_entity:
        entity = predict_entity_name
        if add_indicator and add_doc_name:
            verify_prompt = f' Given a sentence:' \
                f' <doc_name> {doc_name} </doc_name> <sentence> {left_context} <mention> {mention} </mention> {right_context} </sentence> ' \
                f' Please answer "YES" or "NO" that <entity> {entity} {entity_candidate_description} </entity> ' \
                f' best describes the {mention} (identified by <mention> </mention>) in the sentence' \
                f' (identified by <sentence> </sentence>) ?'
        elif add_indicator:
            verify_prompt = f' Given a sentence:' \
                f' <sentence> {left_context} <mention> {mention} </mention> {right_context} </sentence> ' \
                f' Please answer "YES" or "NO" that <entity> {entity} {entity_candidate_description} </entity> ' \
                f' best describes the {mention} (identified by <mention> </mention>) in the sentence' \
                f' (identified by <sentence> </sentence>) ?'
        else:
            if add_doc_name:
                verify_prompt = f' Given a sentence:' \
                    f' {doc_name} {left_context}{mention}{right_context}' \
                    f' Please answer "YES" or "NO" that {entity} {entity_candidate_description}' \
                    f' best describes the {mention} in the sentence ?'
            else:
                verify_prompt = f' Given a sentence:' \
                    f' {left_context}{mention}{right_context}' \
                    f' Please answer "YES" or "NO" that {entity} {entity_candidate_description}' \
                    f' best describes the {mention} in the sentence ?'
        verify_prompt_result = openai_chatgpt(augmented_context_prompt, model=verify_openai_model)

        out_dict['ori_predict_entity'] = out_dict['predict_entity_name']
        out_dict['verify_prompt'] = verify_prompt
        out_dict['verify_prompt_result'] = verify_prompt_result
        if predict_entity_name == '':
            pass
        elif mention.lower() == predict_entity_name.lower():
            pass
        elif 'no' in verify_prompt_result.lower():
            out_dict['predict_entity_name'] = ''
        else:
            pass

    return out_dict


# mention,
# left_context,
# right_context,
# doc_name='',

# rel_args=None
# blink_args=None
add_doc_name=True
add_indicator=True
verify_entity=False
max_rel_entity_candidates=6
max_blink_entity_candidates=3
add_augmented_context=True
add_false_option=True
augment_context_openai_model='gpt-4o'
multi_choice_openai_model='gpt-4o'
verify_openai_model='gpt-4o'
num_context_characters=150
num_entity_description_characters=150


'''
rel_args = prepare_rel_args()
blink_args = prepare_blink_args()
'''

# datasets = ["KORE50", "ace2004", "oke-2015", "oke-2016", "aquaint", "msnbc", "Reuters-128", "RSS-500"]
datasets = ["KORE50"]
input_dir = '/scratch365/yding4/In_Context_EL/RUN_FILES/11_14_2024/ED'
output_dir = '/scratch365/yding4/In_Context_EL/RUN_FILES/11_14_2024/baselines/llm4ed'
os.makedirs(output_dir, exist_ok=True)

for dataset in datasets:
    print('dataset: ', dataset)
    input_file = os.path.join(input_dir, dataset + '.json')
    output_file = os.path.join(output_dir, dataset + '.json')
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)
    if os.path.isfile(output_file):
        with open(output_file) as reader:
            doc_name2instance = json.load(reader)

    for doc_name, instance in tqdm(doc_name2instance.items()):
        if 'out_dicts' in instance:
            continue
        sentence = instance['sentence']
        entities = instance['entities']
        out_dicts = []
        pred_entities = deepcopy(entities)
        for tmp_index, (start, end) in enumerate(zip(
            entities['starts'], entities['ends'],
        )):
            mention = sentence[start: end]
            left_context = sentence[max(0, start - num_context_characters): start]
            right_context = sentence[end: end + num_context_characters]
            out_dict = llm4ed(
                mention,
                left_context,
                right_context,
                doc_name=doc_name,
                rel_args=rel_args,
                blink_args=blink_args,
                add_doc_name=add_doc_name,
                add_indicator=add_indicator,
                verify_entity=verify_entity,
                add_augmented_context=add_augmented_context,
                add_false_option=add_false_option,
                max_rel_entity_candidates=max_rel_entity_candidates,
                max_blink_entity_candidates=max_blink_entity_candidates,
                augment_context_openai_model=augment_context_openai_model,
                multi_choice_openai_model=multi_choice_openai_model,
                verify_openai_model=verify_openai_model,
                num_context_characters=num_context_characters,
                num_entity_description_characters=num_entity_description_characters,
            )

            pred_entities['entity_names'][tmp_index] = out_dict['predict_entity_name']
            out_dicts.append(out_dict)

        doc_name2instance[doc_name]['pred_entities'] = pred_entities
        doc_name2instance[doc_name]['out_dicts'] = out_dicts
        with open(output_file, 'w') as writer:
            json.dump(doc_name2instance, writer, indent=4)




