import blink.main_dense as main_dense
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
from tqdm import tqdm
from in_context_el.dataset_reader import dataset_loader

original_entity2blink_entity = {
    'Lujaizui': 'Lujiazui',
    'Ministry of Defense and Armed Forces Logistics (Iran)': 'Ministry of Defence and Armed Forces Logistics (Iran)',
    'Netzarim (settlement)': 'Netzarim',
    'The Bank of Tokyo-Mitsubishi UFJ': 'MUFG Bank',
    'Time Warner': 'WarnerMedia',
    'Sanford Bernstein':'AllianceBernstein',
    'Capital Cities Communications': 'Capital Cities/ABC Inc.',
    'Reader\'s Digest Association':'Trusted Media Brands',
    'Sprint Nextel': 'Sprint Corporation',
    'John Corbett (actor)':'John Corbett',
    'Electoral College (United States)': 'United States Electoral College',
    'Bob Hope Airport': 'Hollywood Burbank Airport',
}

models_path = "/nfs/yding4/EL_project/BLINK/models/" # the path where you stored the BLINK models

config = {
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

args = argparse.Namespace(**config)

# models = main_dense.load_models(args, logger=None)

(
    biencoder,
    biencoder_params,
    crossencoder,
    crossencoder_params,
    candidate_encoding,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
    faiss_indexer,
) = models

data_to_link = [ {
                    "id": 0,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "".lower(),
                    "mention": "Shakespeare".lower(),
                    "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
                },
                {
                    "id": 1,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "Shakespeare's account of the Roman general".lower(),
                    "mention": "Julius Caesar".lower(),
                    "context_right": "'s murder by his friend Brutus is a meditation on duty.".lower(),
                },
                {
                    "id": 2,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "Tallahassee (United States) 11-15 (AFP) -  The ".lower(),
                    "mention": "Supreme Court".lower(),
                    "context_right": " in Florida today Wednesday refused the application by the state's authorities to stop the new hand count of votes in some counties.".lower(),
                },
                {
                    "id": 3,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "Richard Levin, the Chancellor of this prestigious university, said Talbot would head the ".lower(),
                    "mention": "Globalization Studies Center".lower(),
                    "context_right": " as of next July and would also teach at the university.   \n".lower(),
                },
                {
                    "id": 4,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "".lower(),
                    "mention": "David".lower(),
                    "context_right": " is most likely referring to David Beckham, but without further context or clarification, it cannot be confirmed.".lower(),
                },


                ]

_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)


# 1. load dataset, 

"""
input_file = "/nfs/yding4/e2e_EL_evaluate/data/wned/xml/ori_xml2revise_xml/ace2004/ace2004.xml"
mode = "xml"

num_context_characters = 150
max_num_entity_candidates = 10
doc_name2instance = dataset_loader(input_file, mode=mode)
unknown_entities = []
for doc_name, instance in tqdm(doc_name2instance.items()):
    sentence = instance['sentence']
    entities = instance['entities']
    entity_candidates_list = []
    entity_candidates_description_list = []
    new_entity_names = []
    
    for (
        start,
        end,
        entity_mention,
        entity_name,
    ) in zip(
        entities['starts'],
        entities['ends'],
        entities['entity_mentions'],
        entities['entity_names'],
    ):
        # blink_entity_candidates = []
        if entity_name in original_entity2blink_entity:
            new_entity_name = original_entity2blink_entity[entity_name]
        else:
            new_entity_name = entity_name
        new_entity_names.append(new_entity_name)
        if new_entity_name not in title2id:
            unknown_entities.append(new_entity_name)

        left_context = sentence[max(0, start - num_context_characters): start]
        right_context = sentence[end: end + num_context_characters]
        data_to_link = [ 
            {
                "id": 0,
                "label": "unknown",
                "label_id": -1,
                "context_left": left_context,
                "mention": entity_mention,
                "context_right": right_context,
            },
        ]
        _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
        entity_candidates = predictions[0][:max_num_entity_candidates]
        entity_candidates_description = [id2text[title2id[entity_candidate]] for entity_candidate in entity_candidates]
        entity_candidates_list.append(entity_candidates)
        entity_candidates_description_list.append(entity_candidates_description)

    doc_name2instance[doc_name]['entities']['entity_names'] = new_entity_names
    doc_name2instance[doc_name]['entities']['entity_candidates_list'] = entity_candidates_list
    doc_name2instance[doc_name]['entities']['entity_candidates_description_list'] = entity_candidates_description_list


output_file = 'ace2004.json'
with open(output_file, 'w') as writer:
    json.dump(doc_name2instance, writer, indent=4)
"""

