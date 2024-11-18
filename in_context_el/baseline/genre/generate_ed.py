import os
import json
import pickle
import argparse
from tqdm import tqdm
from in_context_el.dataset_reader import dataset_loader

from genre.fairseq_model import GENRE
from genre.trie import Trie

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
        "--genre_dir",
        help="the parent directory for gendre model and data",
        # required=True,
        default="/nfs/yding4/GENRE_project/GENRE/scripts_genre",
        type=str,
    )
    parser.add_argument(
        "--context_window",
        help="context window to prepare genre input",
        # required=True,
        default=128,
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/4_7_2023/genre",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="output file",
        # required=True,
        default="KORE50.json",
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
    # refined = Refined.from_pretrained(
    #     model_name='wikipedia_model',
    #     entity_set="wikipedia",
    # )

    # load the prefix tree (trie)
    gendre_trie_file = os.path.join(args.genre_dir, "data/kilt_titles_trie_dict.pkl")
    with open(gendre_trie_file, "rb") as f:
        trie = Trie.load_from_dict(pickle.load(f))

    # load the model
    gendre_model_file = os.path.join(args.genre_dir, "models/fairseq_entity_disambiguation_aidayago")
    model = GENRE.from_pretrained(gendre_model_file).eval()


    output_file = args.output_file

    for doc_name, instance in tqdm(doc_name2instance.items()):
        sentence = instance['sentence']
        entities = instance['entities']

        predict_entity_names = []
        for (
            start,
            end,
            entity_mention,
        ) in zip(
            entities['starts'],
            entities['ends'],
            entities['entity_mentions'],
        ):
            precessed_sentence = sentence[
                max(0, start - args.context_window): start
                ] + ' [START_ENT] ' + sentence[start: end]+ ' [END_ENT] ' + sentence[
                    end: end + args.context_window
                    ]
            precessed_sentence = precessed_sentence.replace('  ', ' ')
            output = model.sample(
                sentences=[precessed_sentence],
                prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            )
            if len(output[0]) == 0:
                predict_entity_name = ''
            predict_entity_name = output[0][0]['text']
            if predict_entity_name is None:
                predict_entity_name = ''
            predict_entity_names.append(predict_entity_name)

        doc_name2instance[doc_name]['entities']['predict_entity_names'] = predict_entity_names

    with open(output_file, 'w') as writer:
        json.dump(doc_name2instance, writer, indent=4)
    

if __name__ == '__main__':
    main()
