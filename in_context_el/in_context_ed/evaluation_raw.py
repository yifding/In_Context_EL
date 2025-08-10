import os
import re
import json
import argparse
from REL.wikipedia import Wikipedia
from in_context_el.elastic_matching import elastic_matching


def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluation of 2nd step to collect multi-choice prompt for entity disambiguation.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/entity_candidate_prompt/ace2004.json",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/evaluation",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="output file",
        # required=True,
        default="ace2004",
        type=str,
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
    """Legacy wrapper for elastic_matching function."""
    return elastic_matching(multi_choice_prompt_result, entity_candidates)


def evaluate_ed_chatgpt_multi_choice(args, doc_name2instance):
    num_pred_instance = 0
    num_gt_instance = 0
    num_true_positive = 0
    gold_true_positve = 0

    for doc_name, instance in doc_name2instance.items():
        entities = instance['entities']

        # ground truth
        entity_names = entities['entity_names']

        # chatgpt output
        if 'multi_choice_prompt_results' not in entities:
            continue
        multi_choice_prompt_results = entities['multi_choice_prompt_results']

        # entity candidates
        entity_candidates_list = entities['entity_candidates']

        processed_entity_names = []
        predict_entity_names = []
        for (
            entity_name,
            entity_candidates,
            multi_choice_prompt_result,
        ) in zip(
            entity_names,
            entity_candidates_list,
            multi_choice_prompt_results,
        ):
            # the entity name may have different format (disambiguation, etc)
            processed_entity_name = entity_name.replace('_', ' ')
            predict_entity_name = process_multi_choice_prompt(multi_choice_prompt_result, entity_candidates)

            processed_entity_names.append(processed_entity_name)
            predict_entity_names.append(predict_entity_name)

            if processed_entity_name != '':
                num_gt_instance += 1
                if processed_entity_name in entity_candidates:
                    gold_true_positve += 1
                else:
                    print(f'processed_entity_name: {processed_entity_name}')
                    print(f'entity_candidates: {entity_candidates}')

                if predict_entity_name == '':
                    continue
                else:
                    num_pred_instance += 1
                    if predict_entity_name == processed_entity_name:
                        num_true_positive += 1

        # print(f'entity_candidates: {entity_candidates}')
        # print(f'predict_entity_names: {predict_entity_names}')
        # print(f'processed_entity_names: {processed_entity_names}')
        # break
        doc_name2instance[doc_name]['entities']['processed_entity_names'] = processed_entity_names
        doc_name2instance[doc_name]['entities']['predict_entity_names'] = predict_entity_names

    precision = dev_by_zero(num_true_positive, num_pred_instance)
    recall = dev_by_zero(num_true_positive, num_gt_instance)
    f1 = dev_by_zero(2*precision*recall, precision + recall)

    gold_recall = dev_by_zero(gold_true_positve, num_gt_instance)

    out_dict = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gold_recall': gold_recall,
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
    out_dict = evaluate_ed_chatgpt_multi_choice(args, doc_name2instance)
    print(out_dict)

if __name__ == '__main__':
    main()