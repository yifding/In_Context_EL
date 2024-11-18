import os
import json
import argparse
import requests
from tqdm import tqdm
from in_context_el.dataset_reader import dataset_loader


def parse_args():
    parser = argparse.ArgumentParser(
        description='1st step to collect prompt for entity information.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        help="the extension file used by load_dataset function to load dataset",
        # required=True,
        choices=["tsv", "oke_2015", "oke_2016", "n3", "xml"],
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
        default="/nfs/yding4/In_Context_EL/RUN_FILES/4_7_2023/end2end_neural_el/generate_ed",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="output file",
        # required=True,
        default="KORE50.json",
        type=str,
    )
    parser.add_argument(
        "--URL",
        default="http://localhost:5555",
        type=str,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    assert os.path.isfile(args.input_file)
    return args


def main():
    args = parse_args()
    doc_name2instance = dataset_loader(args.input_file, key=args.key, mode=args.mode)
    output_file = args.output_file

    for doc_name, instance in tqdm(doc_name2instance.items()):
        sentence = instance['sentence']
        entities = instance['entities']

        predict_entity_names = []
        span_list = []
        for (
            start,
            end,
            entity_mention,
        ) in zip(
            entities['starts'],
            entities['ends'],
            entities['entity_mentions'],
        ):
            span = {"start":start,"length": end - start}
            span_list.append(span)

        myjson = { "text": sentence, "spans": span_list}
        spans = eval(requests.post(args.URL, json=myjson).content.decode("utf-8"))
        start2entity = dict()
        for span in spans:
            start = span[0]
            entity = span[2]
            start2entity[start] = entity

        for start in entities['starts']:
            if start in start2entity:
                entity = start2entity[start]
                predict_entity_names.append(entity)
            else:
                predict_entity_names.append("")
        
        doc_name2instance[doc_name]['entities']['predict_entity_names'] = predict_entity_names

    with open(output_file, 'w') as writer:
        json.dump(doc_name2instance, writer, indent=4)
    

if __name__ == '__main__':
    main()
