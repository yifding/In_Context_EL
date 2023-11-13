import os
import jsonlines
from tqdm import tqdm


def read_type_hierarchy(old_hier):
    train_types = {}
    reverse_train_types = {}
    with open(old_hier) as f:
        for line in f:
            if (len(line.strip().split(':')) < 2):
                continue
            tmp = line.strip().split(':')
            # if tmp[0] not in train_type_count:
            #     continue
            train_types[tmp[0]] = tmp[1]
            reverse_train_types[tmp[1]] = tmp[0].split('/')[1:]
    print('\ntrain_types:')
    print(train_types)
    print('\nreverse_train_types:')
    print(reverse_train_types)
    return train_types, reverse_train_types


old_hier = '/afs/crc.nd.edu/user/y/yding4/ET_project/dataset/bbn/bbn_types.txt'
train_types, reverse_train_types = read_type_hierarchy(old_hier)

# input_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/10_25_2023/blink_candidates/bbn_test_results.json'
input_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/10_25_2023/blink_candidates/bbn_test_one_step_results.json'

with jsonlines.open(input_file) as reader:
    records = [record for record in reader]

pred_instances = 0
gt_instance = 0
true_positive = 0
for record_index, record in enumerate(records):
    # partial evaluation
    # if 'prompt_answer' not in record or record_index==5000:
    if 'prompt_answer' not in record:
        print(f'{record_index + 1} of {len(records)} have been evaluated!')
        break
    prompt_answer = record['prompt_answer']
    y_category = record['y_category']
    y_prediction = set()
    for train_type in reverse_train_types:
        if train_type not in prompt_answer:
            continue
        prompt_answer_type = prompt_answer[train_type]
        if prompt_answer_type.lower().startswith('yes'):
            raw_types = reverse_train_types[train_type]
            for tmp_index_raw_type, raw_type in enumerate(raw_types):
                processed_types = '/' + '/'.join(raw_types[:tmp_index_raw_type+1])
                y_prediction.add(processed_types)

    gt_instance += len(y_category)
    pred_instances += len(y_prediction)
    for y_prediction_ele in y_prediction:
        if y_prediction_ele in y_category:
            true_positive += 1

precision = true_positive / pred_instances if pred_instances > 0 else 0
recall = true_positive / gt_instance if gt_instance > 0 else 0
f1 = 2 * precision*recall / (precision + recall) if (precision + recall) > 0 else 0

print(f'gt_instance:{gt_instance}; pred_instances:{pred_instances}; true_positive:{true_positive}')
print(f'precision: {precision}; recall: {recall}; f1: {f1};')