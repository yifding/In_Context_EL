import os
import json
from tqdm import tqdm
from refined.inference.processor import Refined


refined = Refined.from_pretrained(model_name='aida_model', entity_set="wikipedia")

# spans = refined.process_text("England won the FIFA World Cup in 1966.")


# load datasets
input_dataset_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/blink_candidates'
output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/baseline/refined/prediction'
os.makedirs(output_dir, exist_ok=True)
datasets = ['KORE50', 'ace2004', 'aida_test', 'aquaint', 'clueweb', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500', 'wikipedia']
for dataset in datasets:
    print('dataset:', dataset)
    input_dataset_file = os.path.join(input_dataset_dir, dataset + '.json')
    output_file = os.path.join(output_dir, dataset + '.json')
    with open(input_dataset_file) as reader:
        doc_name2instance = json.load(reader)

    for doc_name, instance in tqdm(doc_name2instance.items()):
        if 'blink_entity_candidates_list' in instance['entities']:
            del doc_name2instance[doc_name]['entities']['blink_entity_candidates_list']

        sentence = instance['sentence']
        spans = refined.process_text(sentence)
        # print('sentence:', sentence)
        # print('spans:', spans)
        # print('\n')
        pred_entities = {
            'starts': [],
            'ends': [],
            'entity_mentions': [],
            'entity_names': [],
        }
        for span in spans:
            start = span.start
            mention = str(span.text)
            end = start + len(mention)
            entity = span.predicted_entity.wikipedia_entity_title
            if entity == '' or entity is None:
                entity = ''
            pred_entities['starts'].append(start)
            pred_entities['ends'].append(end)
            pred_entities['entity_mentions'].append(mention)
            pred_entities['entity_names'].append(entity)
        doc_name2instance[doc_name]['pred_entities'] = pred_entities
    
    with open(output_file, 'w') as writer:
        json.dump(doc_name2instance , writer, indent=4)


