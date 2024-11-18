import os
import json
from tqdm import tqdm

input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/baseline/refined/prediction'
datasets = ['derczynski', 'KORE50', 'RSS-500']
output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/baseline/refined/prediction_test_token_shift'
os.makedirs(output_dir, exist_ok=True)

for dataset in datasets:
    print(f'dataset: {dataset}')
    input_file = os.path.join(input_dir, dataset + '.json')
    output_file = os.path.join(output_dir, dataset + '.json')
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)
    

    for doc_name, instance in tqdm(doc_name2instance.items()):
        sentence = instance['sentence']
        pred_entities = instance['pred_entities']

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
                            print(f'sentence: {sentence[max(0, start - 100): start]} <Mention>{entity_mention}</Mention> {sentence[end: end + 100]}')
                            print(f'original: start: {start}; end: {end}; mention: {entity_mention};')

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
    
    with open(output_file, 'w') as reader:
        json.dump(doc_name2instance, reader, indent=4)
