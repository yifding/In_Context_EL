import os
import json


def base_preprocess(s):
        if s is None or s == '' or s.lower() == 'none' or s.lower() == 'nil':
            return ''
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


def process_with_rel(entity_name, wikipedia):
    init_entity_name = base_preprocess(entity_name)
    if init_entity_name == '':
        return init_entity_name
    processed_entity_name = wikipedia.preprocess_ent_name(init_entity_name)
    return base_preprocess(processed_entity_name).lower()


def process_with_refined(entity_name, wikidata_mapper):
    if entity_name is None or entity_name == '' or entity_name.lower() == 'none' or entity_name.lower() == 'nil':
            return ''
    
    init_entity_name = entity_name.replace(" ", "_")
    qcode = wikidata_mapper.map_title_to_wikidata_qcode(init_entity_name)
    if qcode is None or wikidata_mapper.wikidata_qcode_is_disambiguation_page(qcode):
        # print(f'entity_name: {entity_name} ')
        return ''
    else:
        return str(qcode)


def process_entity_name(entity_name, wikipedia=None, wikidata_mapper=None):
    if wikipedia is not None:
        processed_entity_name = process_with_rel(entity_name, wikipedia)
    elif wikidata_mapper is not None:
        processed_entity_name = process_with_refined(entity_name, wikidata_mapper)
    else:
        processed_entity_name = base_preprocess(entity_name).lower()
    return processed_entity_name


def obtain_set_entities(entities, wikipedia=None, wikidata_mapper=None, keep_none=False):
    assert 'starts' in entities
    assert 'ends' in entities 
    assert 'entity_names' in entities
    re_set_entities = set()
    re_set_index = set()
    zero_set_entities = set()
    for (start, end, entity_name) in zip(
        entities['starts'],
        entities['ends'],
        entities['entity_names'],
    ):
        processed_entity_name = process_entity_name(entity_name, wikipedia, wikidata_mapper)
        if processed_entity_name == '':
            if keep_none and entity_name is not None and entity_name != '' and entity_name.lower() != 'none' and entity_name.lower() != 'nil':
                element = (start, end, 'Q1')
                if (start, end) in re_set_index:
                    pass
                else:
                    re_set_index.add((start, end))
                    re_set_entities.add(element)
            else:
                zero_set_entities.add((start, end, ''))
            continue

        element = (start, end, processed_entity_name)
        if (start, end) in re_set_index:
            continue
        else:
            re_set_index.add((start, end))
        # assert element not in re_set_entities
        re_set_entities.add(element)
    return re_set_entities, zero_set_entities


def evaluate_doc_name2instance(doc_name2instance, wikipedia=None, wikidata_mapper=None, keep_none=False):

    total_pred = 0
    total_gt = 0
    tp = 0
    for doc_name in doc_name2instance:
        instance = doc_name2instance[doc_name]
        entities = instance['entities']
        pred_entities = instance['pred_entities']
        gt_set_entities, gt_zero_set_entities = obtain_set_entities(entities, wikipedia, wikidata_mapper, keep_none=keep_none)
        pred_set_entities, _ = obtain_set_entities(pred_entities, wikipedia, wikidata_mapper)

        total_gt += len(gt_set_entities)
        total_pred += len(pred_set_entities)

        for pred_set_entity in pred_set_entities:
            if pred_set_entity in gt_set_entities:
                tp += 1
            else:
                start, end = pred_set_entity[0], pred_set_entity[1]
                if (start, end, '') in gt_zero_set_entities:
                    total_pred -= 1
    
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
    # input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_verify_processed'
    # input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_update_verify_processed'
    
    
    # datasets = ['msnbc', 'KORE50', 'oke_2015', 'oke_2016']
    # datasets = ['oke_2015']
    
    # datasets = ['KORE50', 'RSS-500', 'derczynski']
    
    # input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_update_part3_verify_processed'
    # datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500', 'aida_test', 'derczynski']

    input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/baseline/refined/ED_standard_datasets/prediction'
    datasets = ['ace2004', 'aquaint', 'msnbc', 'aida_test', 'clueweb', 'wikipedia']

    use_rel = True
    use_refined = False
    wikipedia = None
    wikidata_mapper = None
    keep_none = False    # whether to keep all the non-empty entities from the datasets
    if use_rel:
        from REL.wikipedia import Wikipedia
        base_url = '/nfs/yding4/REL/data/'
        wiki_version = 'wiki_2019'
        # wikipedia = Wikipedia(args.base_url, args.wiki_version)
        wikipedia = Wikipedia(base_url, wiki_version)
    elif use_refined:
        from refined.resource_management.aws import S3Manager
        from refined.resource_management.resource_manager import ResourceManager
        from refined.doc_preprocessing.wikidata_mapper import WikidataMapper

        data_dir = os.path.join(os.path.expanduser('~'), '.cache', 'refined')
        datasets_dir = os.path.join(data_dir, 'datasets')
        additional_data_dir = os.path.join(data_dir, 'additional_data')

        resource_manager = ResourceManager(
            S3Manager(),
            data_dir=data_dir,
            datasets_dir=datasets_dir,
            additional_data_dir=additional_data_dir,
            entity_set=None,
            model_name=None,
        )
        wikidata_mapper = WikidataMapper(resource_manager=resource_manager)

    for dataset in datasets:
        print('dataset:', dataset)
        input_file = os.path.join(input_dir, dataset + '.json')
        output_file = os.path.join(input_dir, dataset + '_metrics.json')
        with open(input_file) as reader:
            doc_name2instance = json.load(reader)
            out_dict = evaluate_doc_name2instance(doc_name2instance, wikipedia, wikidata_mapper, keep_none=keep_none)
            with open(output_file, 'w') as writer:
                json.dump(out_dict, writer, indent=4)