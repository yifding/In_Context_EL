import os
import json
from REL.wikipedia import Wikipedia

from in_context_el.baseline.evaluate import (
    evaluate_doc_name2instance,
    obtain_set_entities,
)


def compare_set_entities(
    pred_set_entities, 
    gt_set_entities, 
    gt_zero_set_entities,
    sentence,
    reverse_order=False, 
    add_mention_identifier=True,
    left_mention_identifier = '<Mention>',
    right_mention_identifier = '</Mention>',
    context_num_char=150,
):
    set_entity2overlap = []
    for pred_set_entity in pred_set_entities:
        (pred_start, pred_end, pred_entity) = pred_set_entity
        if not reverse_order:
            if pred_set_entity in gt_set_entities or (pred_start, pred_end, '') in gt_zero_set_entities:
                continue
        else:
            if pred_set_entity in gt_set_entities:
                continue
        FLAG = False
        pred_mention = sentence[pred_start: pred_end]
        left_context = sentence[max(0, pred_start - context_num_char): pred_start]
        right_context = sentence[pred_end: pred_end + context_num_char]
        if add_mention_identifier:
            pred_s = left_context + left_mention_identifier + pred_mention + right_mention_identifier + right_context
        else:
            pred_s = left_context + pred_mention + right_context
        for gt_set_entity in gt_set_entities:
            (gt_start, gt_end, gt_entity) = gt_set_entity
            if pred_start <= gt_start < pred_end or gt_start <= pred_start < gt_end:
                FLAG = True
                gt_mention = sentence[gt_start: gt_end]
                left_context = sentence[max(0, gt_start - context_num_char): gt_start]
                right_context = sentence[gt_end: gt_end + context_num_char]
                if add_mention_identifier:
                    gt_s = left_context + left_mention_identifier + gt_mention + right_mention_identifier + right_context
                else:
                    gt_s = left_context + gt_mention + right_context

                tmp_dict = {
                    'cur_entity_set': f'{pred_start}::::{pred_end}::::{pred_mention}::::{pred_entity}',
                    'rel_entity_set': f'{gt_start}::::{gt_end}::::{gt_mention}::::{gt_entity}',
                    'cur_sentence': pred_s,
                    'rel_sentence': gt_s,
                    'same_start': 1 if gt_start == pred_start else 0,
                    'same_end': 1 if gt_end == pred_end else 0,
                    'same_both': 1 if gt_start == pred_start and gt_end == pred_end else 0,
                    'same_entity': 1 if gt_entity == pred_entity else 0,
                }
                set_entity2overlap.append(tmp_dict)
        
        if not FLAG:
            tmp_dict = {
                'cur_entity_set': f'{pred_start}::::{pred_end}::::{pred_mention}::::{pred_entity}',
                'cur_sentence': pred_s,
            }
            set_entity2overlap.append(tmp_dict)

    return set_entity2overlap


# input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/prediction'
# input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_verify_processed'
# input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_update_part3_verify_processed'
# datasets = ['aida_test']
# datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500', 'aida_test', 'derczynski']
input_dir = '/nfs/yding4/In_Context_EL/in_context_el/end2end_model_agent/models/prompt_engineering/first/'
datasets = ['KORE50', 'aquaint']
# datasets = ['ace2004', 'aquaint', 'msnbc', 'aida_test', 'clueweb']


base_url = '/nfs/yding4/REL/data/'
wiki_version = 'wiki_2019'
wikipedia = Wikipedia(base_url, wiki_version)
context_num_char = 150
add_mention_identifier = True
left_mention_identifier = '<Entity>'
right_mention_identifier = '</Entity>'




for dataset in datasets:
    print('\ndataset:', dataset)
    input_file = os.path.join(input_dir, dataset + '.json')
    false_positive_json = os.path.join(input_dir, dataset +'_fp.json')
    false_negative_json = os.path.join(input_dir, dataset + '_fn.json')
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)

    doc_name2fp = dict()
    doc_name2fn = dict()
    for doc_index, doc_name in enumerate(doc_name2instance):
        # print('doc_name:', doc_name)

        instance = doc_name2instance[doc_name]
        sentence = instance['sentence']
        entities = instance['entities']
        pred_entities = instance['pred_entities']
        gt_set_entities, gt_zero_set_entities = obtain_set_entities(entities, wikipedia)
        pred_set_entities, pred_zero_set_entities = obtain_set_entities(pred_entities, wikipedia)

        gt_set_entities = sorted(gt_set_entities)
        gt_zero_set_entities = sorted(gt_zero_set_entities)
        pred_set_entities = sorted(pred_set_entities)
        pred_zero_set_entities = sorted(pred_zero_set_entities)

        # if doc_name == 'chtb_165.eng':
        #     print(f'pred_set_entities: {pred_set_entities}')
        #     print(f'pred_zero_set_entities: {pred_zero_set_entities}')
        #     print(f'gt_set_entities: {gt_set_entities}')
        #     print(f'gt_zero_set_entities: {gt_zero_set_entities}')

        # false positive, may change to a function later
        set_entity2overlap = compare_set_entities(
            pred_set_entities, 
            gt_set_entities, 
            gt_zero_set_entities,
            sentence, 
            reverse_order=False, 
            add_mention_identifier=True,
            left_mention_identifier=left_mention_identifier,
            right_mention_identifier=right_mention_identifier,
            context_num_char=context_num_char,
        )
        doc_name2fp[doc_name] = set_entity2overlap

        revserse_set_entity2overlap = compare_set_entities(
            gt_set_entities, 
            pred_set_entities, 
            pred_zero_set_entities,
            sentence, 
            reverse_order=True, 
            add_mention_identifier=True,
            left_mention_identifier=left_mention_identifier,
            right_mention_identifier=right_mention_identifier,
            context_num_char=context_num_char,
        )
        doc_name2fn[doc_name] = revserse_set_entity2overlap
    
    with open(false_positive_json, 'w') as writer:
        json.dump(doc_name2fp, writer, indent=4)
    
    with open(false_negative_json, 'w') as writer:
        json.dump(doc_name2fn, writer, indent=4)




'''
        ori_instance = ori_doc_name2instance[doc_name]
        ori_pred_entities = ori_instance['pred_entities']
        ori_pred_set_entities = obtain_set_entities(ori_pred_entities, wikipedia)

        for ori_pred_set_entity in sorted(ori_pred_set_entities):
            if ori_pred_set_entity not in gt_set_entities:
                continue
            ori_start, ori_end, ori_entity = ori_pred_set_entity
            for pred_set_entity in pred_set_entities:
                start, end, entity = pred_set_entity
                if ori_start == start and ori_end == end and entity != ori_entity:
                    sent_start = max(0, start-50)
                    sent_end = end + 50 
                    select_sentence = sentence[sent_start: sent_end]
                    print('gt_entity:', ori_pred_set_entity)
                    print('pred_entity:', pred_set_entity)
                    print('select_sentence:', select_sentence)
                    print('\n\n')
                    count += 1
    print('count:', count)
'''