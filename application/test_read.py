import os
import csv
from tqdm import tqdm
import pandas as pd
import blink.main_dense as main_dense

# 0. processing input data
def to_records(input_file):
    df = pd.read_csv(input_file, sep='\t')
    tmp_records = df.to_records(index=False)
    records = []
    keys = list(df.keys())
    for tmp_record in tmp_records:
        record = {}
        assert len(tmp_record) == len(keys)
        for tmp_index, key in enumerate(keys):
            record[key] = tmp_record[tmp_index]
        records.append(record)
    return records

input_file = 'should-the-us-adopt-stricter-gun-controls-3346.csv'
records = to_records(input_file)

# 1. prepare data for entity candidate extraction and 
max_num_entity_candidates = 10
num_context_characters = 150
for tmp_record_index, record in tqdm(enumerate(records)):
    if tmp_record_index == 0:
        continue
    print(tmp_record_index)
    record = records[0]
    sentence = record['sentence']
    ARG0 = record['ARG0']
    ARG1 = record['ARG1']

    print(f'ARG0: {ARG0}; sentence: {sentence}')
    assert ARG0 in sentence
    print(f'ARG1: {ARG1}; sentence: {sentence}')
    assert ARG1 in sentence
    ARG0_start = sentence.index(ARG0)
    ARG0_end = ARG0_start + len(ARG0)
    ARG1_start = sentence.index(ARG1)
    ARG1_end = ARG1_start + len(ARG1)


# 2. load BLINK model
models_path = '/nfs/yding4/EL_project/BLINK/models/' # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": args.blink_num_candidates,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}

# blink_args = argparse.Namespace(**config)

# models = main_dense.load_models(blink_args, logger=None)

# data_to_link = [ 
#     {
#         "id": 0,
#         "label": "unknown",
#         "label_id": -1,
#         "context_left": sentence[max(0, ARG0_start - num_context_characters): ARG0_start],
#         "mention": ARG0,
#         "context_right": sentence[ARG0_end: ARG0_end + num_context_characters],
#     },
#     {
#         "id": 0,
#         "label": "unknown",
#         "label_id": -1,
#         "context_left": sentence[max(0, ARG1_start - num_context_characters): ARG1_start],
#         "mention": ARG1,
#         "context_right": sentence[ARG1_end: ARG1_end + num_context_characters],
#     },
# ]

# _, _, _, _, _, predictions, scores, = main_dense.run(blink_args, None, *models, test_data=data_to_link)
# ARG0_entity_candidates = predictions[0][:max_num_entity_candidates]
# ARG1_entity_candidates = predictions[1][:max_num_entity_candidates]




