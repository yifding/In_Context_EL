import os
import json
import argparse
from tqdm import tqdm
import blink.main_dense as main_dense
from in_context_el.dataset_reader import dataset_loader
from in_context_el.original_entity2blink_entity import original_entity2blink_entity

# use cpu by default
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description='1st step to collect prompt for entity information.',
        allow_abbrev=False,
    )

    # dataset loader parameters
    parser.add_argument(
        "--mode",
        help="the extension file used by load_dataset function to load dataset",
        # required=True,
        choices=["tsv", "oke_2015", "oke_2016", "n3", "xml", "unseen_mentions"],
        default="xml",
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

    # I/O parameters
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        # required=True,
        default="/nfs/yding4/e2e_EL_evaluate/data/wned/xml/ori_xml2revise_xml/ace2004/ace2004.xml",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/3_28_2023/mention_prompt",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="output file",
        # required=True,
        default="ace2004.json",
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
    parser.add_argument(
        "--num_entity_description_characters",
        help="maximum number of characters allowed for each entity candidate",
        # required=True,
        default=150,
        type=int,
    )
    parser.add_argument(
        "--max_num_entity_candidates",
        help="maximum number of entity candidates of each mention",
        # required=True,
        default=10,
        type=int,
    )

    parser.add_argument(
        "--openai_mode",
        help="",
        # required=True,
        default='chatgpt',
        choices=['chatgpt', 'gpt'],
        type=str,
    )
    parser.add_argument(
        "--openai_model",
        help="",
        # required=True,
        default='gpt-3.5-turbo',
        choices=['gpt-3.5-turbo', 'text-curie-001', 'text-davinci-003'],
        type=str,
    )

    parser.add_argument(
        "--blink_models_path",
        help="the dataset file used by load_dataset to load dataset, must end with / ",
        # required=True,
        default="/nfs/yding4/EL_project/BLINK/models/",
        type=str,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    assert os.path.isfile(args.input_file)
    return args

def main():
    args = parse_args()
    blink_models_path = args.blink_models_path

    blink_config = {
        "test_entities": None,
        "test_mentions": None,
        "interactive": False,
        "top_k": args.max_num_entity_candidates,
        "biencoder_model": blink_models_path+"biencoder_wiki_large.bin",
        "biencoder_config": blink_models_path+"biencoder_wiki_large.json",
        "entity_catalogue": blink_models_path+"entity.jsonl",
        "entity_encoding": blink_models_path+"all_entities_large.t7",
        "crossencoder_model": blink_models_path+"crossencoder_wiki_large.bin",
        "crossencoder_config": blink_models_path+"crossencoder_wiki_large.json",
        "fast": False, # set this to be true if speed is a concern
        "output_path": "logs/" # logging directory
    }

    blink_args = argparse.Namespace(**blink_config)

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


    # 1. load dataset,
    input_file = args.input_file
    mode = args.mode

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
            _, _, _, _, _, predictions, scores, = main_dense.run(blink_args, None, *models, test_data=data_to_link)
            entity_candidates = predictions[0][:max_num_entity_candidates]
            entity_candidates_description = [id2text[title2id[entity_candidate]] for entity_candidate in entity_candidates]
            entity_candidates_list.append(entity_candidates)
            entity_candidates_description_list.append(entity_candidates_description)

        doc_name2instance[doc_name]['entities']['entity_names'] = new_entity_names
        doc_name2instance[doc_name]['entities']['entity_candidates_list'] = entity_candidates_list
        doc_name2instance[doc_name]['entities']['entity_candidates_description_list'] = entity_candidates_description_list

    output_file = args.output_file
    with open(output_file, 'w') as writer:
        json.dump(doc_name2instance, writer, indent=4)

if __name__ == '__main__':
    main()
