from tqdm import tqdm
import argparse
import jsonlines
import blink.main_dense as main_dense

import torch
torch.cuda.set_device(1)

input_file = '/nfs/yding4/In_Context_EL/data/et/huang/bbn/bbn_test.json'
output_file = '/nfs/yding4/In_Context_EL/in_context_el/in_context_et/bbn_test.json'
with jsonlines.open(input_file) as reader:
    records = [record for record in reader] 

# for record in records:
#     if record['mention_as_list'] == []:
#         print(record)
        # print(record['mention_as_list'])

# 1. load BLINK model
models_path = "/nfs/yding4/EL_project/BLINK/models/" # the path where you stored the BLINK models

blink_config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}

blink_args = argparse.Namespace(**blink_config)
models = main_dense.load_models(blink_args, logger=None)


for record_index, record in tqdm(enumerate(records)):
    left_context = record['left_context_text']
    right_context = record['right_context_text']
    mention = record['word']
    data_to_link = [ 
        {
            "id": 0,
            "label": "unknown",
            "label_id": -1,
            "context_left": left_context,
            "mention": mention,
            "context_right": right_context,
        },
    ]
    _, _, _, _, _, predictions, scores, = main_dense.run(blink_args, None, *models, test_data=data_to_link)
    entity_candidates = predictions[0][:10]
    records[record_index]['entity_candidates'] = entity_candidates


with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(records)