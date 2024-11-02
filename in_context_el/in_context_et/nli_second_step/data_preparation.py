# load csv file, generate entity_types, generate entity_instances

import os
import csv
import json
import random
import jsonlines
from tqdm import tqdm
import jsonlines

random.seed(19940802)


def process_dbpedia_class_name(s):
    re_s = ''
    for s_ele_index, s_ele in enumerate(s):
        if s_ele.isupper():
            if s_ele_index < len(s) - 1 and s[s_ele_index+1].isupper():
                re_s += s_ele
            elif s_ele_index == len(s) - 1:
                re_s += s_ele
            elif re_s == '':
                re_s += s_ele.lower()
            else:
                re_s += ' ' + s_ele.lower()
        else:
            re_s += s_ele
    return re_s


input_dir = '/afs/crc.nd.edu/user/y/yding4/ET_project/dataset/dbpedia'
output_dir = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/11_2_2023/nli_second_step'
os.makedirs(output_dir, exist_ok=True)
full_file = os.path.join(input_dir, 'DBP_wiki_data.csv')
train_file = os.path.join(input_dir, 'DBPEDIA_train.csv')
dev_file = os.path.join(input_dir, 'DBPEDIA_val.csv')
test_file = os.path.join(input_dir, 'DBPEDIA_test.csv')
output_fine2ultrafine = os.path.join(output_dir, 'fine2ultrafine.json')
output_general2fine = os.path.join(output_dir, 'general2fine.json')
output_full_file = os.path.join(output_dir, 'full_file.json')
output_train_file = os.path.join(output_dir, 'train_file.json')
output_dev_file = os.path.join(output_dir, 'dev_file.json')
output_test_file = os.path.join(output_dir, 'test_file.json')


general2fine = dict()
fine2ultrafine = dict()

# collect hierarchy, dbpedia has three-level entity labels
# ['text', 'l1', 'l2', 'l3', 'wiki_name', 'word_count']
# records = []
tmp_length = 6
with open(full_file) as reader:
    spamreader = csv.reader(reader)
    for line_index, row in enumerate(tqdm(spamreader)):
        if line_index == 0:
            tmp_length = len(row)
            continue
        if len(row) != tmp_length:
            continue
        record = dict()
        record['text'] = row[0]
        record['l1'] = process_dbpedia_class_name(row[1])
        record['l2'] = process_dbpedia_class_name(row[2])
        record['l3'] = process_dbpedia_class_name(row[3])
        record['wiki_name'] = row[4]

        if record['l1'] not in general2fine:
            general2fine[record['l1']] = []
        if record['l2'] not in general2fine[record['l1']]:
            general2fine[record['l1']].append(record['l2'])
        if record['l2'] not in fine2ultrafine:
            fine2ultrafine[record['l2']] = []
        if record['l3'] not in fine2ultrafine[record['l2']]:
            fine2ultrafine[record['l2']].append(record['l3'])
#
with open(output_fine2ultrafine, 'w') as writer:
    json.dump(fine2ultrafine, writer, indent=4)
with open(output_general2fine, 'w') as writer:
    json.dump(general2fine, writer, indent=4)

# process train/dev/test data into jsonlines
for input_json_file, output_json_file in zip(
    [train_file, dev_file, test_file],
    [output_train_file, output_dev_file, output_test_file],
):
    records = []
    with open(input_json_file) as reader:
        csv_reader = csv.reader(reader)
        for line_index, row in enumerate(tqdm(csv_reader)):
            if line_index == 0:
                tmp_length = len(row)
                continue
            if len(row) != tmp_length:
                continue
            record = dict()
            record['text'] = row[0]
            record['l1'] = process_dbpedia_class_name(row[1])
            record['l2'] = process_dbpedia_class_name(row[2])
            record['l3'] = process_dbpedia_class_name(row[3])
            # record['wiki_name'] = row[4]
            records.append(record)
    with jsonlines.open(output_json_file, 'w') as writer:
        writer.write_all(records)

