import os
import re
import json
import argparse
from tqdm import tqdm
from in_context_el.openai_function import openai_chatgpt


def process_multi_choice_prompt(multi_choice_prompt_result, entity_candidates):

    L = len(entity_candidates)
    if L == 0:
        return ''
    elif L == 1:
        return entity_candidates[0]
    elif 'None of the entity match' in multi_choice_prompt_result:
        return ''
    # palm may return non type for results
    if type(multi_choice_prompt_result) is not str:
        return ''

    if any (s in multi_choice_prompt_result.lower() for s in [' not ', 'doesn\'t', 'none']):
        return ''
    
    # update the index finding schema with regular expression.
    
    index_list = [int(s) - 1 for s in re.findall(r'\b\d+\b', multi_choice_prompt_result) if 0 <= int(s) - 1 < len(entity_candidates)]
    # index_list = []
    # for index in range(L):
    #     if str(index + 1) in multi_choice_prompt_result:
    #         index_list.append(index)
    
    # consider direct index answer of chatgpt
    if len(index_list) == 1:
        return entity_candidates[index_list[0]]
    
    # if there are two choices and candidate entities length is more than 2, select the first one.
    if len(index_list) == 2 and len(entity_candidates) > 2:
        return entity_candidates[index_list[0]]

    # consider complete string match
    index_list = []
    for index, entity_candidate in enumerate(entity_candidates):
        if entity_candidate.lower() in multi_choice_prompt_result.lower():
            add_flag = True
            other_candidates = entity_candidates[:index] + entity_candidates[index+1:]
            for other_candidate in other_candidates:
                if entity_candidate.lower() in other_candidate.lower() and other_candidate.lower() in multi_choice_prompt_result.lower():
                    add_flag = False
                    break
            if add_flag:
                index_list.append(index)

    if len(index_list) ==1:
        # print(index_list[0])
        # print(len(entity_candidates))
        return entity_candidates[index_list[0]]
    
    return ''


def LLM_verify_full(mention, left_context, right_context, entity, entity_description=None, exact_match_release=True):
    prompt = f'''
        Given a sentence: \n
        <sentence> {left_context} <mention> {mention} </mention> {right_context} </sentence> \n
        Please answer "YES" or "NO" that <entity> {entity} </entity> and <entity_description> {entity_description} </entity_description> \n
        best describes the <mention> {mention} </mention> mentioned in the sentence ?
    '''

    prompt_response = openai_chatgpt(prompt, model='gpt-4o')
    
    if 'yes' in prompt_response.lower():
        prompt_result = 1
    else:
        prompt_result = 0
    if exact_match_release and mention.lower() == entity.lower():
        prompt_result = 1
    return prompt, prompt_response, prompt_result


def LLM4context_augment(mention, left_context, right_context):
    context_prompt = f'''
    Given a sentence:\n
    <sentence> {left_context} <mention> {mention} </mention> {right_context} </sentence> \n
    What does <mention> {mention} </mention> in this sentence referring to ?
    '''
    context_prompt_response = openai_chatgpt(context_prompt, model='gpt-4o')
    return context_prompt, context_prompt_response


def blink_w_context(
        mention, 
        left_context, 
        right_context, 
        title2id,   # from blink
        id2text,    # from blink
        blink_args, # from blink
        main_dense, # from blink
        models, # from blink
        k=10, # default parameter for blink
    ):
        data_to_link = [ 
            {
                "id": 0,
                "label": "unknown",
                "label_id": -1,
                "context_left": left_context,
                "mention": mention,
                "context_right": right_context,
            },
        ]

        _, _, _, _, _, predictions, _, = main_dense.run(blink_args, None, *models, test_data=data_to_link)
        entity_candidates = predictions[0][:k]
        entity_candidates_descriptions = []
        for entity_candidate in entity_candidates:
            text = id2text[title2id[entity_candidate]]
            entity_candidates_descriptions.append(text)
        
        return entity_candidates, entity_candidates_descriptions


def LLM4EntGPT_P():
    pass


def LLM4ED_selection(mention, context, entity_candidates, entity_candidates_descriptions, num_entity_description_characters=150):
    assert len(entity_candidates) == len(entity_candidates_descriptions)
    multi_choice_prompt = ''
    for index, (entity_candidate, entity_candidate_description) in enumerate(zip(entity_candidates, entity_candidates_descriptions)):
        description = f'<entity> {entity_candidate} </entity>' + ' ' + f'<description> {entity_candidate_description[:num_entity_description_characters]} <description>'
        multi_choice_prompt += f'({index + 1}). ' + description + '\n'

    multi_choice_prompt = context + '\n\n' + f'Which of the following does <mention> {mention} </mention> refer to?' + '\n\n' + multi_choice_prompt
    multi_choice_prompt_response = openai_chatgpt(multi_choice_prompt, model='gpt-4o')
    entity = process_multi_choice_prompt(multi_choice_prompt_response, entity_candidates)
    return entity, multi_choice_prompt, multi_choice_prompt_response


def LLM4ED(
        mention, 
        left_context, 
        right_context, 
        title2id,   # from blink
        id2text,    # from blink
        blink_args, # from blink
        main_dense, # from blink
        models, # from blink
        k=10, 
        num_context_characters=150, 
        num_entity_description_characters=150):
    # 1. LLM4context_augment
    # 2. blink with augmented context
    # 3. LLM4entity_selection

    # 1. LLM4context_augment
    context_prompt, context_prompt_response = LLM4context_augment(mention, left_context, right_context)

    if mention in context_prompt_response:
        start = context_prompt_response.index(mention)
        end = start + len(mention)
        left_context = context_prompt_response[max(0, start - num_context_characters): start]
        right_context = context_prompt_response[end: end + num_context_characters]   

    # 2. blink with augmented context
    entity_candidates, entity_candidates_descriptions = blink_w_context(
        mention, 
        left_context, 
        right_context, 
        title2id,   # from blink
        id2text,    # from blink
        blink_args, # from blink
        main_dense, # from blink
        models, # from blink
        k=k,
    )

    # 3. LLM4entity_selection
    entity, multi_choice_prompt, multi_choice_prompt_response = LLM4ED_selection(
        mention, 
        context_prompt_response, 
        entity_candidates, 
        entity_candidates_descriptions, 
        num_entity_description_characters=num_entity_description_characters,
    )
    if entity in entity_candidates:
        index = entity_candidates.index(entity)
        entity_candidates_description = entity_candidates_descriptions[index]
    else:
        entity_candidates_description = ''

    return {
        'entity': entity,
        'entity_candidates_description': entity_candidates_description,
        'context_prompt': context_prompt,
        'context_prompt_response': context_prompt_response,
        'multi_choice_prompt': multi_choice_prompt,
        'multi_choice_prompt_response': multi_choice_prompt_response,
        'entity_candidates': entity_candidates,
        'entity_candidates_descriptions': entity_candidates_descriptions,
    }
    

def apply_LLM_verify_full():
    
    # '''
    # input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/prediction'
    # output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM_verify_full'
    input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED'
    output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_verify'
    os.makedirs(output_dir, exist_ok=True)
    # datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500']
    datasets = ['aida_test']
    from REL.wikipedia import Wikipedia 
    base_url = '/nfs/yding4/REL/data/'
    wiki_version = 'wiki_2019'
    wikipedia = Wikipedia(base_url, wiki_version)
    # '''
    # datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500']
    # '''

    max_context_character = 150
    for dataset in datasets:
        print(dataset)
        input_file = os.path.join(input_dir, dataset + '.json')
        output_file = os.path.join(output_dir, dataset + '.json')
        if os.path.isfile(output_file):
            input_file = output_file
        with open(input_file) as reader:
            doc_name2instance = json.load(reader)
        
        for doc_name, instance in tqdm(doc_name2instance.items()):
            if 'LLM_verify_full' in instance['pred_entities']:
                continue
            sentence = instance['sentence']
            pred_entities = instance['pred_entities']
            instance['pred_entities']['LLM_verify_full'] = {
                'prompt': [],
                'prompt_response': [],
                'prompt_result': [],
            }
            for (start, end, entity_name) in zip(
                pred_entities['starts'],
                pred_entities['ends'],
                pred_entities['entity_names'],
                ):
                mention = sentence[start: end]
                left_context = sentence[max(0, start - max_context_character) :start]
                right_context = sentence[end: end + max_context_character]
                processed_entity_name = wikipedia.preprocess_ent_name(entity_name)
                prompt, prompt_response, prompt_result = LLM_verify_full(mention, left_context, right_context, processed_entity_name)
                
                instance['pred_entities']['LLM_verify_full']['prompt'].append(prompt)
                instance['pred_entities']['LLM_verify_full']['prompt_response'].append(prompt_response)
                instance['pred_entities']['LLM_verify_full']['prompt_result'].append(prompt_result)
            doc_name2instance[doc_name] = instance
            with open(output_file, 'w') as writer:
                json.dump(doc_name2instance, writer, indent=4)
    # '''


def apply_LLM4ED():
    # datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500']

    # load blink
    import blink.main_dense as main_dense
    import torch
    torch.cuda.set_device(1)

    # '''
    blink_num_candidates = 100
    models_path = "/nfs/yding4/EL_project/BLINK/models/"

    config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": blink_num_candidates,
            "biencoder_model": models_path+"biencoder_wiki_large.bin",
            "biencoder_config": models_path+"biencoder_wiki_large.json",
            "entity_catalogue": models_path+"entity.jsonl",
            "entity_encoding": models_path+"all_entities_large.t7",
            "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
            "crossencoder_config": models_path+"crossencoder_wiki_large.json",
            "fast": False, # set this to be true if speed is a concern
            "output_path": "logs/" # logging directory
    }

    blink_args = argparse.Namespace(**config)

    models = main_dense.load_models(blink_args, logger=None)

    (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = models
    # '''

    # datasets = ['KORE50']
    datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500', 'aida_test']
    input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/prediction'
    output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED'
    os.makedirs(output_dir, exist_ok=True)
    k = 10
    max_context_characters = 150
    num_entity_description_characters = 150
    for dataset in datasets:
        print(dataset)
        input_file = os.path.join(input_dir, dataset + '.json')
        output_file = os.path.join(output_dir, dataset + '.json')
        if os.path.isfile(output_file):
            input_file = output_file

        with open(input_file) as reader:
            doc_name2instance = json.load(reader)

        for doc_name, instance in tqdm(doc_name2instance.items()):
            if 'out_dicts' in instance:
                continue
            sentence = instance['sentence']
            pred_entities = instance['pred_entities']
            new_pred_entities = []
            out_dicts = []
            for start, end in zip(pred_entities['starts'], pred_entities['ends']):
                mention = sentence[start: end]
                left_context = sentence[max(0, start - max_context_characters) :start]
                right_context = sentence[end: end + max_context_characters]
                out_dict = LLM4ED(
                    mention, 
                    left_context, 
                    right_context, 
                    title2id,   # from blink
                    id2text,    # from blink
                    blink_args, # from blink
                    main_dense, # from blink
                    models, # from blink
                    k=k, 
                    num_context_characters=max_context_characters,
                    num_entity_description_characters=num_entity_description_characters,
                )
                out_dicts.append(out_dict)
                new_pred_entity = out_dict['entity']
                new_pred_entities.append(new_pred_entity)
            
            doc_name2instance[doc_name]['pred_entities']['entity_names'] = new_pred_entities
            doc_name2instance[doc_name]['out_dicts'] = out_dicts

            with open(output_file, 'w') as writer:
                json.dump(doc_name2instance, writer, indent=4)


if __name__ == '__main__':
    apply_LLM_verify_full()
    # apply_LLM4ED()



