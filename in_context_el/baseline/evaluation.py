import os
import re
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='direct evaluation for entity prediction.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/4_7_2023/refined/generate_ed/KORE50.json",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/4_7_2023/refined/evaluation",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="output file",
        required=True,
        default="KORE50",
        type=str,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.dataset + '_metric.json')
    assert os.path.isfile(args.input_file)
    return args

def dev_by_zero(a, b):
    if b == 0:
        return 0.0
    else:
        return a / b


def process_entity(e):
    return e.replace('_', ' ').lower()


def evaluate_ed(doc_name2instance):
    num_pred_instance = 0
    num_gt_instance = 0
    num_true_positive = 0

    for doc_name, instance in doc_name2instance.items():
        entities = instance['entities']

        # ground truth
        entity_names = entities['entity_names']

        # prediction
        predict_entity_names = entities['predict_entity_names']

        for (
            entity_name,
            predict_entity_name,
        ) in zip(
            entity_names,
            predict_entity_names,
        ):
            if entity_name != '':
                num_gt_instance += 1
                if predict_entity_name == '':
                    continue
                else:
                    num_pred_instance += 1
                    if process_entity(predict_entity_name) == process_entity(entity_name):
                        num_true_positive += 1

        # print(f'entity_candidates: {entity_candidates}')
        # print(f'predict_entity_names: {predict_entity_names}')
        # print(f'processed_entity_names: {processed_entity_names}')
        # break

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

    return out_dict


def main():
    args = parse_args()
    input_file = args.input_file
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)

    out_dict = evaluate_ed(doc_name2instance)
    with open(args.output_file, 'w') as writer:
        json.dump(out_dict, writer, indent=4)


if __name__ == '__main__':
    main()