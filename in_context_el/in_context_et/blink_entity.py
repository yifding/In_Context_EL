import os
import argparse
import jsonlines
from tqdm import tqdm
import blink.main_dense as main_dense

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='1st step to collect conduct entity disambiguation with BLINK.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        # required=True,
        default="/afs/crc.nd.edu/user/y/yding4/ET_project/dataset/bbn/bbn_test.json",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default="/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/10_25_2023/blink_candidates/",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="output file",
        # required=True,
        default="bbn_test.json",
        type=str,
    )

    # blink parameters
    parser.add_argument(
        "--blink_models_path",
        help="blink model path, must ends with /",
        # required=True,
        default="/afs/crc.nd.edu/user/y/yding4/EL_project/BLINK/models/",
        type=str,
    )

    parser.add_argument(
        "--blink_num_candidates",
        help="number of entity candidates for blink model",
        # required=True,
        default=10,
        type=int,
    )

    parser.add_argument(
        "--device",
        help="gpu_index",
        # required=True,
        default=1,
        type=int,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    assert os.path.isfile(args.input_file)
    return args


def main():

    # input_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/dataset/bbn/sample_10_bbn_test.json'
    # output_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/' \
    #               '10_25_2023/blink_candidates/sample_10_bbn_test.json'
    # input_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/dataset/bbn/bbn_test.json'
    # output_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/' \
    #               '10_25_2023/blink_candidates/bbn_test.json'
    args = parse_args()
    torch.cuda.set_device(args.device)
    input_file = args.input_file
    output_file = args.output_file

    with jsonlines.open(input_file) as reader:
        records = [record for record in reader]

    # for record in records:
    #     if record['mention_as_list'] == []:
    #         print(record)
            # print(record['mention_as_list'])

    # 1. load BLINK model
    models_path = "/afs/crc.nd.edu/user/y/yding4/EL_project/BLINK/models/" # the path where you stored the BLINK models

    blink_config = {
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

    for record_index, record in enumerate(tqdm(records)):
        left_context = record['left_context_text']
        right_context = record['right_context_text']
        mention = record['word']
        if len(mention) > 50:
            mention = mention[:50]
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
        entity_candidates_descriptions = []
        for entity_candidate in entity_candidates:
            text = id2text[title2id[entity_candidate]]
            entity_candidates_descriptions.append(text)
        records[record_index]['entity_candidates'] = entity_candidates
        records[record_index]['entity_candidates_descriptions'] = entity_candidates_descriptions

    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(records)


if __name__ == '__main__':
    main()