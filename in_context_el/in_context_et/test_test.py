import os
import jsonlines

input_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/10_25_2023/blink_candidates/bbn_test_one_step_results.json'

gt_set = set()

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
print(reverse_train_types)