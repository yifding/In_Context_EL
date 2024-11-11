import os
import json
from REL.wikipedia import Wikipedia

def preprocess(s):
    s = s.replace('_', ' ').replace('-', ' ').replace('Ã³', 'ó')

    last_space = True
    meet_bracket = False
    output_s = ''
    for s_i in s:
        if last_space and not meet_bracket:
            output_s += s_i.upper()
            last_space = False
        else:
            output_s += s_i
            if s_i == ' ' or s_i == '.':
                last_space = True
        if s_i == '(':
            meet_bracket = True
        if s_i == ')':
            meet_bracket = False

    return output_s


def obtain_set_entities(entities, wikipedia, remove_none=True):
    assert 'starts' in entities
    assert 'ends' in entities 
    assert 'entity_names' in entities
    re_set_entities = set()
    for (start, end, entity_name) in zip(
        entities['starts'],
        entities['ends'],
        entities['entity_names'],
    ):
        if remove_none:
            if entity_name == '' or entity_name == 'None':
                continue
        if wikipedia != None:
            processed_entity_name = wikipedia.preprocess_ent_name(preprocess(entity_name))
            processed_entity_name = preprocess(processed_entity_name).lower()
        else:
            processed_entity_name = preprocess(entity_name).lower()

        element = (start, end, processed_entity_name)
        assert element not in re_set_entities
        re_set_entities.add(element)
    return re_set_entities


def evaluate_doc_name2instance(doc_name2instance, wikipedia=None):

    total_pred = 0
    total_gt = 0
    tp = 0
    for doc_name in doc_name2instance:
        instance = doc_name2instance[doc_name]
        entities = instance['entities']
        pred_entities = instance['pred_entities']
        gt_set_entities = obtain_set_entities(entities, wikipedia)
        pred_set_entities = obtain_set_entities(pred_entities, wikipedia)

        total_gt += len(gt_set_entities)
        total_pred += len(pred_set_entities)

        for pred_set_entity in pred_set_entities:
            if pred_set_entity in gt_set_entities:
                tp += 1
    
    precision = tp / (total_pred + 1e-8)
    recall = tp / (total_gt + 1e-8)
    f1 = 2*precision*recall / (precision + recall + 1e-8)

    out_dict = {
        'true positives': tp,
        'total_pred': total_pred,
        'total_gt': total_gt,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    print(out_dict)
    return out_dict
    
if __name__ == '__main__':
    # input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/prediction'
    # datasets = ['KORE50', 'ace2004', 'aida_test', 'aquaint', 'clueweb', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500', 'wikipedia']
    # input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM_verify_full_processed'
    # datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500']
    # input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED'
    # datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500', 'aida_test']
    input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_verify_processed'
    # datasets = ['msnbc', 'KORE50', 'oke_2015', 'oke_2016']
    datasets = ['aida_test']


    base_url = '/nfs/yding4/REL/data/'
    wiki_version = 'wiki_2019'
    # wikipedia = Wikipedia(args.base_url, args.wiki_version)
    wikipedia = Wikipedia(base_url, wiki_version)

    for dataset in datasets:
        print('dataset:', dataset)
        input_file = os.path.join(input_dir, dataset + '.json')
        output_file = os.path.join(input_dir, dataset + '_metrics.json')
        with open(input_file) as reader:
            doc_name2instance = json.load(reader)
            out_dict = evaluate_doc_name2instance(doc_name2instance, wikipedia)
            with open(output_file, 'w') as writer:
                json.dump(out_dict, writer, indent=4)
