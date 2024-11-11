import os
import json
from REL.wikipedia import Wikipedia

from in_context_el.end2end_model_agent.evaluate import (
    evaluate_doc_name2instance,
    obtain_set_entities,
)


# input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/prediction'
input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_verify_processed'
ori_input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/prediction'
datasets = ['aida_test']

base_url = '/nfs/yding4/REL/data/'
wiki_version = 'wiki_2019'
# wikipedia = Wikipedia(base_url, wiki_version)


for dataset in datasets:
    print('dataset:', dataset)
    input_file = os.path.join(input_dir, dataset + '.json')
    ori_input_file = os.path.join(ori_input_dir, dataset + '.json')
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)

    with open(ori_input_file) as reader:
        ori_doc_name2instance = json.load(reader)

    for doc_index, doc_name in enumerate(doc_name2instance):
        print('doc_name:', doc_name)
        instance = doc_name2instance[doc_name]
        sentence = instance['sentence']
        entities = instance['entities']
        pred_entities = instance['pred_entities']
        gt_set_entities = obtain_set_entities(entities, wikipedia)
        pred_set_entities = obtain_set_entities(pred_entities, wikipedia)

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


        '''
        for pred_set_entity in sorted(pred_set_entities):
            if pred_set_entity not in gt_set_entities:
                start, end, entity = pred_set_entity
                print('mention:', sentence[start: end],'pred_entity:', pred_set_entity)
                sent_start = max(0, start-50)
                sent_end = end + 50
                for gt_set_entity in gt_set_entities:
                    gt_start, gt_end, gt_entity = gt_set_entity
                    if gt_start >= sent_start and gt_end <= sent_end:
                        print('gt_entity:', gt_set_entity)
                select_sentence = sentence[sent_start: sent_end]
                print('select_sentence:', select_sentence)
                print('\n\n')
        '''

        if doc_index == 20:
            break

        

