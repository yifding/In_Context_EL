import os
import json

input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_verify'
output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/LLM4ED_verify_processed'
os.makedirs(output_dir, exist_ok=True)
# datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500']
datasets = ['aida_test']
for dataset in datasets:
    input_file = os.path.join(input_dir, dataset + '.json')
    output_file = os.path.join(output_dir, dataset + '.json')
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)
    for doc_name, instance in doc_name2instance.items():
        pred_entities = instance['pred_entities']
        LLM_verify_full = pred_entities['LLM_verify_full']
        new_pred_entities = {
            'starts': [],
            'ends': [],
            'entity_mentions': [],
            'entity_names': [],
        }

        for (start, end, entity_mention, entity_name, prompt_result) in zip(
            pred_entities['starts'],
            pred_entities['ends'],
            pred_entities['entity_mentions'],
            pred_entities['entity_names'],
            LLM_verify_full['prompt_result'],
        ):
            if prompt_result:
                new_pred_entities['starts'].append(start)
                new_pred_entities['ends'].append(end)
                new_pred_entities['entity_mentions'].append(entity_mention)
                new_pred_entities['entity_names'].append(entity_name)
        
        doc_name2instance[doc_name]['pred_entities'] = new_pred_entities
    
    with open(output_file, 'w') as writer:
        json.dump(doc_name2instance, writer, indent=4)
