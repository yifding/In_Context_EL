import os
import re
import json
import argparse
import blink.main_dense as main_dense

import torch
torch.cuda.set_device(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='2nd step to collect multi-choice prompt for entity disambiguation.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/multi_choice_prompt/KORE50.json",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/evaluation",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="output file",
        # required=True,
        default="KORE50",
        type=str,
    )
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
    assert os.path.isfile(args.input_file)
    return args

def dev_by_zero(a, b):
    if b == 0:
        return 0.0
    else:
        return a / b
        
def process_multi_choice_prompt(multi_choice_prompt_result, entity_candidates):

    L = len(entity_candidates)
    if L == 0:
        return ''
    elif L == 1:
        return entity_candidates[0]
    elif 'None of the entity match' in multi_choice_prompt_result:
        return ''
    
    # update the index finding schema with regular expression.
    
    index_list = [int(s) - 1 for s in re.findall(r'\b\d+\b', multi_choice_prompt_result) if 0 <= int(s) - 1 < len(entity_candidates)]
    # index_list = []
    # for index in range(L):
    #     if str(index + 1) in multi_choice_prompt_result:
    #         index_list.append(index)
    
    # consider direct index answer of chatgpt
    if len(index_list) == 1:
        return entity_candidates[index_list[0]]
    
    # if there are two choices and candidate entities length is more than 2, select the first one.
    if len(index_list) == 2 and len(entity_candidates) > 2:
        return entity_candidates[index_list[0]]

    # consider complete string match
    index_list = []
    for index, entity_candidate in enumerate(entity_candidates):
        if entity_candidate.lower() in multi_choice_prompt_result.lower():
            add_flag = True
            other_candiddates = entity_candidates[:index] + entity_candidates[index+1:]
            for other_candidate in other_candiddates:
                if entity_candidate in other_candidate:
                    add_flag = False
                    break
            if add_flag:
                index_list.append(index)

    if len(index_list) == 1:
        return entity_candidates[index_list[0]]
    
    return ''


def evaluate_ed_chatgpt_multi_choice(doc_name2instance, title2id, args):
    num_pred_instance = 0
    num_gt_instance = 0
    num_true_positive = 0
    gold_true_positve = 0

    for doc_name, instance in doc_name2instance.items():
        entities = instance['entities']

        # ground truth
        entity_names = entities['entity_names']
        tmp_predicted_entity_names = entities['predict_entity_names']
        # chatgpt output
        # multi_choice_prompt_results = entities['multi_choice_prompt_results']

        # entity candidates
        # entity_candidates_list = entities['entity_candidates']

        processed_entity_names = []
        predict_entity_names = []

        for (
            entity_name,
            tmp_predict_entity_name,

        ) in zip(
            entity_names,
            tmp_predicted_entity_names
        ):
            # the entity name may have different format (disambiguation, etc)
            processed_entity_name = entity_name
            predict_entity_name = tmp_predict_entity_name

            processed_entity_names.append(processed_entity_name)
            predict_entity_names.append(predict_entity_name)

            if processed_entity_name != '' and processed_entity_name in title2id:
                num_gt_instance += 1
                if predict_entity_name == '':
                    continue
                else:
                    num_pred_instance += 1
                    if predict_entity_name == processed_entity_name:
                        num_true_positive += 1

        doc_name2instance[doc_name]['entities']['processed_entity_names'] = processed_entity_names
        doc_name2instance[doc_name]['entities']['predict_entity_names'] = predict_entity_names

    precision = dev_by_zero(num_true_positive, num_pred_instance)
    recall = dev_by_zero(num_true_positive, num_gt_instance)
    f1 = dev_by_zero(2*precision*recall, precision + recall)

    out_dict = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_true_positive': num_true_positive,
        'num_pred_instance': num_pred_instance,
        'num_gt_instance': num_gt_instance,
    }

    with open(os.path.join(args.output_dir, args.dataset + '.json'), 'w') as writer:
        json.dump(doc_name2instance, writer, indent=4)

    with open(os.path.join(args.output_dir, args.dataset + '_metric.json'), 'w') as writer:
            json.dump(out_dict, writer, indent=4)
    
    return out_dict


def main():
    args = parse_args()

    input_file = args.input_file
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)

    models_path = args.blink_models_path  # the path where you stored the BLINK models

    config = {
        "test_entities": None,
        "test_mentions": None,
        "interactive": False,
        "top_k": args.blink_num_candidates,
        "biencoder_model": models_path + "biencoder_wiki_large.bin",
        "biencoder_config": models_path + "biencoder_wiki_large.json",
        "entity_catalogue": models_path + "entity.jsonl",
        "entity_encoding": models_path + "all_entities_large.t7",
        "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
        "crossencoder_config": models_path + "crossencoder_wiki_large.json",
        "fast": False,  # set this to be true if speed is a concern
        "output_path": "logs/"  # logging directory
    }

    blink_args = argparse.Namespace(**config)

    # load blink model
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

    out_dict = evaluate_ed_chatgpt_multi_choice(doc_name2instance, title2id, args)
    print(out_dict)


if __name__ == '__main__':
    main()