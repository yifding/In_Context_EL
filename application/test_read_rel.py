import os
import csv
from tqdm import tqdm
import jsonlines
import pandas as pd
import requests

IP_ADDRESS = "http://localhost"
PORT = "5555"


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


input_file = '/nfs/yding4/In_Context_EL/data/ed/wanying/news_sample_srl_output.csv'
output_file = '/nfs/yding4/In_Context_EL/data/ed/wanying/news_sample_srl_output_with_entities.jsonl'
records = to_records(input_file)

new_records = []
# 1. prepare data for entity candidate extraction and
for record_index, record in enumerate(tqdm(records)):
    sentence = record['sentence']
    document = {
        "text": sentence,
        "spans": [],  # in case of ED only, this can also be left out when using the API
    }
    API_result = requests.post("{}:{}".format(IP_ADDRESS, PORT), json=document).json()
    new_record = {
        'sentence': sentence,
        'entities': API_result,
    }
    new_records.append(new_record)

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

with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(new_records)
