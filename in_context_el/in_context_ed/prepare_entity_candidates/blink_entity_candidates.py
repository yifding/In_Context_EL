import blink.main_dense as main_dense
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import jsonlines
from tqdm import tqdm
from in_context_el.dataset_reader import dataset_loader

import torch
torch.cuda.set_device(1)

# original_entity2blink_entity = {
#     'Lujaizui': 'Lujiazui',
#     'Ministry of Defense and Armed Forces Logistics (Iran)': 'Ministry of Defence and Armed Forces Logistics (Iran)',
#     'Netzarim (settlement)': 'Netzarim',
#     'The Bank of Tokyo-Mitsubishi UFJ': 'MUFG Bank',
#     'Time Warner': 'WarnerMedia',
#     'Sanford Bernstein':'AllianceBernstein',
#     'Capital Cities Communications': 'Capital Cities/ABC Inc.',
#     'Reader\'s Digest Association':'Trusted Media Brands',
#     'Sprint Nextel': 'Sprint Corporation',
#     'John Corbett (actor)':'John Corbett',
#     'Electoral College (United States)': 'United States Electoral College',
#     'Bob Hope Airport': 'Hollywood Burbank Airport',
# }

def parse_args():
    parser = argparse.ArgumentParser(
        description='1st step to collect blink entity candidates for entity disambiguation.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        help="the extension file used by load_dataset function to load dataset",
        # required=True,
        choices=["jsonl", "tsv", "oke_2015", "oke_2016", "n3", "xml", "unseen_mentions"],
        default="tsv",
        type=str,
    )
    parser.add_argument(
        "--key",
        help="the split key of aida-conll dataset",
        # required=True,
        choices=["", "testa", "testb"],
        default="",
        type=str,
    )
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        # required=True,
        default="/nfs/yding4/EL_project/dataset/KORE50/AIDA.tsv",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/mention_prompt",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="output file",
        # required=True,
        default="KORE50.json",
        type=str,
    )
    # hyper parameters:
    parser.add_argument(
        "--num_context_characters",
        help="maximum number of characters of original input sentence around mention",
        # required=True,
        default=150,
        type=int,
    )

    # blink parameters
    parser.add_argument(
            "--blink_models_path",
            help="blink model path, must ends with /",
            # required=True,
            default="/nfs/yding4/EL_project/BLINK/models/",
            type=str,
        )
    
    parser.add_argument(
            "--blink_num_candidates",
            help="number of entity candidates for blink model",
            # required=True,
            default=10,
            type=int,
        )
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    assert os.path.isfile(args.input_file)
    return args


def main():
    args = parse_args()
    models_path = args.blink_models_path # the path where you stored the BLINK models

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

    blink_args = argparse.Namespace(**config)

    models = main_dense.load_models(blink_args, logger=None)

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

    """
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
    _, _, _, _, _, predictions, scores, = main_dense.run(blink_args, None, *models, test_data=data_to_link)
    """

    # 1. load dataset, 

    input_file = args.input_file
    mode = args.mode

    num_context_characters = 150
    max_num_entity_candidates = 10
    if mode == 'jsonl':
        doc_name2instance = dict()
        with jsonlines.open(input_file) as reader:
            for record in reader:
                doc_name = record.pop('doc_name')
                doc_name2instance[doc_name] = record
    else:
        doc_name2instance = dataset_loader(input_file, mode=mode)

    for doc_name, instance in tqdm(doc_name2instance.items()):
        sentence = instance['sentence']
        entities = instance['entities']
        entity_candidates_list = []
        
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
            _, _, _, _, _, predictions, scores, = main_dense.run(blink_args, None, *models, test_data=data_to_link)
            entity_candidates = predictions[0][:max_num_entity_candidates]
            entity_candidates_list.append(entity_candidates)

        doc_name2instance[doc_name]['entities']['blink_entity_candidates_list'] = entity_candidates_list

    output_file = args.output_file
    with open(output_file, 'w') as writer:
        json.dump(doc_name2instance, writer, indent=4)


if __name__ == '__main__':
    main()
