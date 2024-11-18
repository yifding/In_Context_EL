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
    # **YD** injecting special characters confuse BLINK model to obtain entity candidates. Taking them out.
    # context_prompt = f'''
    # Given a sentence:\n
    # <sentence> {left_context} <mention> {mention} </mention> {right_context} </sentence> \n
    # What does <mention> {mention} </mention> in this sentence referring to ?
    # '''

    # https://github.com/yifding/In_Context_EL/blob/main/in_context_el/in_context_ed/first_step_prompt.py#L53
    context_prompt = left_context + mention + right_context + " \n What does " + mention + " in this sentence referring to?"

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


def LLM4ED_selection(mention, context, entity_candidates, entity_candidates_descriptions, num_entity_description_characters=150, model='gpt-4o'):
    assert len(entity_candidates) == len(entity_candidates_descriptions)

    # **YD** adding special tokens may not be well recognized by entgpt-i
    # multi_choice_prompt = ''
    # for index, (entity_candidate, entity_candidate_description) in enumerate(zip(entity_candidates, entity_candidates_descriptions)):
    #     description = f'<entity> {entity_candidate} </entity>' + ' ' + f'<description> {entity_candidate_description[:num_entity_description_characters]} <description>'
    #     multi_choice_prompt += f'({index + 1}). ' + description + '\n'

    # multi_choice_prompt = context + '\n\n' + f'Which of the following does <mention> {mention} </mention> refer to?' + '\n\n' + multi_choice_prompt
    
    # EntGPT-P
    # https://github.com/yifding/In_Context_EL/blob/main/in_context_el/in_context_ed/second_step_prompt.py#L61

    # EntGPT-P: instruction tuning
    # https://github.com/yifding/In_Context_EL/blob/main/in_context_el/in_context_ed/one_step_prompt/intruction_tuning_one_step_openai.py#L139
    multi_choice_prompt = ''
    dummy_entity = 'None of the entity match.'
    entity_candidates = entity_candidates[:9]
    entity_candidates_descriptions = entity_candidates_descriptions[:9]
    entity_candidates.append(dummy_entity)
    entity_candidates_descriptions.append('')
    for index, (entity_candidate, entity_candidate_description) in enumerate(zip(entity_candidates, entity_candidates_descriptions)):
        description = entity_candidate + ' ' + entity_candidate_description[:num_entity_description_characters]
        multi_choice_prompt += f'({index + 1}). ' + description + '\n'

    # multi_choice_prompt = context + '\n\n' + f'Which of the following does {mention} refer to?' + '\n\n' + multi_choice_prompt
    multi_choice_prompt = context + ' ' + f'Which of the following entities is {mention} in this sentence?' + ' ' + multi_choice_prompt

    multi_choice_prompt_response = openai_chatgpt(multi_choice_prompt, model=model)
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
        num_entity_description_characters=150,
        model='gpt-4o',
        ):
    # 1. LLM4context_augment
    # 2. blink with augmented context
    # 3. LLM4entity_selection

    # 1. LLM4context_augment
    context_prompt, context_prompt_response = LLM4context_augment(mention, left_context, right_context)

    if mention in context_prompt_response:
        start = context_prompt_response.index(mention)
        end = start + len(mention)
        prompt_left_context = context_prompt_response[max(0, start - num_context_characters): start]
        promopt_right_context = context_prompt_response[end: end + num_context_characters]   
    else:
        prompt_left_context = left_context
        promopt_right_context = right_context

    # 2. blink with augmented context
    entity_candidates, entity_candidates_descriptions = blink_w_context(
        mention, 
        prompt_left_context, 
        promopt_right_context, 
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
        # **YD** use original context for entity selection
        # context_prompt_response, 
        left_context + mention + right_context,
        entity_candidates, 
        entity_candidates_descriptions, 
        num_entity_description_characters=num_entity_description_characters,
        model=model,
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
    # input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_update'
    # output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_update_verify_full'
    # input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED'
    # output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_verify'
    # input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/prediction'
    # output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM_verify_full'
    input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_update_part3'
    output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_update_part3_verify_full'
    os.makedirs(output_dir, exist_ok=True)
    # datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500']
    # datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500', 'aida_test']
    datasets = ['KORE50', 'derczynski', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500', 'aida_test']
    from REL.wikipedia import Wikipedia 
    base_url = '/nfs/yding4/REL/data/'
    wiki_version = 'wiki_2019'
    wikipedia = Wikipedia(base_url, wiki_version)

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


def apply_LLM4ED(
    models_path = "/nfs/yding4/EL_project/BLINK/models/",
    datasets = ['derczynski'],
    input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/baseline/refined/prediction',
    output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_update_part3',
    model='ft:gpt-3.5-turbo-0613:amrit::8VNXmmdS',
    shift_token_boundary=True,
):

    # load blink
    import blink.main_dense as main_dense
    import torch
    torch.cuda.set_device(1)

    # '''
    blink_num_candidates = 10
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
    # datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500', 'aida_test']
    # datasets = ['oke_2015']
    
    os.makedirs(output_dir, exist_ok=True)
    k = 10
    max_context_characters = 150
    num_entity_description_characters = 150
    for dataset in datasets:
        print(dataset)
        if shift_token_boundary and dataset in ['derczynski', 'KORE50', 'RSS-500']:
            shift_token_boundary = True
        else:
            shift_token_boundary = False
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

            # if 'pred_entities' in instance: for second part of the EL
            # else: for complete inference for ED
            if 'pred_entities' not in instance:
                instance['pred_entities'] = instance['entities']
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
                    # model='gpt-4o',
                    model=model,
                )
                out_dicts.append(out_dict)
                new_pred_entity = out_dict['entity']
                new_pred_entities.append(new_pred_entity)
            
            if shift_token_boundary:
                new_starts = []
                new_ends = []
                new_mentions = []
                tokens = sentence.split(' ')
                tmp_start = 0
                tmp_starts = []
                tmp_ends = []
                for token_index, token in enumerate(tokens):
                    tmp_starts.append(tmp_start)
                    tmp_ends.append(tmp_start + len(token))
                    tmp_start += 1 + len(token)

                for start, end, entity_mention in zip(pred_entities['starts'], pred_entities['ends'], pred_entities['entity_mentions']):
                    if start in tmp_starts and end in tmp_ends:
                        new_starts.append(start)
                        new_ends.append(end)
                        new_mentions.append(sentence[start:end])
                        assert sentence[start:end] == entity_mention
                        continue
                    if start not in tmp_starts:
                        for second_index, (second_start, second_end) in enumerate(zip(tmp_starts, tmp_ends)):
                            if start == second_end:
                                if second_index != len(tmp_starts) - 1:
                                    start = tmp_starts[second_index + 1]
                                    break
                                else:
                                    raise ValueError('in the end')
                            elif second_start < start < second_end:
                                start = second_start
                                break
                            
                    if end not in tmp_ends:
                        for second_index, (second_start, second_end) in enumerate(zip(tmp_starts, tmp_ends)):
                            if end == second_start:
                                if second_index != 0:
                                    end = tmp_ends[second_index - 1]
                                    break
                                else:
                                    raise ValueError('in the beginning')
                            elif second_start < end < second_end:
                                end = second_end
                                break

                    new_starts.append(start)
                    new_ends.append(end)
                    new_mentions.append(sentence[start:end])


                doc_name2instance[doc_name]['pred_entities']['starts'] = new_starts
                doc_name2instance[doc_name]['pred_entities']['ends'] = new_ends
                doc_name2instance[doc_name]['pred_entities']['entity_mentions'] = new_mentions

            doc_name2instance[doc_name]['pred_entities']['entity_names'] = new_pred_entities
            doc_name2instance[doc_name]['out_dicts'] = out_dicts

            with open(output_file, 'w') as writer:
                json.dump(doc_name2instance, writer, indent=4)



if __name__ == '__main__':
    # apply_LLM_verify_full()
    # apply_LLM4ED()
    models_path = "/nfs/yding4/EL_project/BLINK/models/"
    # datasets = ['derczynski'],
    datasets = ['ace2004', 'aquaint', 'msnbc', 'aida_test', 'clueweb', 'wikipedia']
    input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/ED_standard_datasets'
    output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/baseline/LLM4ED/ED_standard_datasets/prediction/'
    model='ft:gpt-3.5-turbo-0613:amrit::8VNXmmdS'
    shift_token_boundary=False

    apply_LLM4ED(
        models_path = models_path,
        datasets = datasets,
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        shift_token_boundary=shift_token_boundary,
    )
