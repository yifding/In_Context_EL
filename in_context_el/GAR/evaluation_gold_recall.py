import os
import re
import json
import argparse
# import REL
from REL.wikipedia import Wikipedia


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
    parser.add_argument(
        "--entity_candidate_name",
        help="key for entity candidate name",
        default="entity_candidates",
        type=str,
    )

    parser.add_argument(
        "--base_url",
        help="data directory for REL",
        default="/nfs/yding4/REL/data/",
        type=str,
    )

    parser.add_argument(
        "--wiki_version",
        help="data directory for REL",
        default="wiki_2019",
        choices=["wiki_2014", "wiki_2019"],
        type=str,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    assert os.path.isfile(args.input_file)
    return args

def preprocess(s):
    s = s.replace('_', ' ').replace('-', ' ').replace('Ã³', 'ó')

    last_space = True
    meet_bracket = False
    output_s = ''
    for s_i in s:
        if last_space and not meet_bracket:
            output_s += s_i.upper()
            last_space = False
        else:
            output_s += s_i
            if s_i == ' ' or s_i == '.':
                last_space = True
        if s_i == '(':
            meet_bracket = True
        if s_i == ')':
            meet_bracket = False

    return output_s


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
    # palm may return non type for results
    if type(multi_choice_prompt_result) is not str:
        return ''

    if any (s in multi_choice_prompt_result.lower() for s in [' not ', 'doesn\'t', 'none']):
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
            other_candidates = entity_candidates[:index] + entity_candidates[index+1:]
            for other_candidate in other_candidates:
                if entity_candidate.lower() in other_candidate.lower() and other_candidate.lower() in multi_choice_prompt_result.lower():
                    add_flag = False
                    break
            if add_flag:
                index_list.append(index)

    if len(index_list) ==1:
        # print(index_list[0])
        # print(len(entity_candidates))
        return entity_candidates[index_list[0]]
    
    return ''


def evaluate_ed_chatgpt_multi_choice(args, doc_name2instance):
    num_pred_instance = 0
    num_gt_instance = 0
    num_true_positive = 0
    gold_true_positve = 0
    top_1 = 0
    top_3 = 0
    top_5 = 0
    top_10 = 0

    for doc_name, instance in doc_name2instance.items():
        # print('b')
        entities = instance['entities']

        # ground truth
        entity_names = entities['entity_names']


        # entity candidates
        # entity_candidates_list = entities['blink_entity_candidates_list']
        # entity_candidates_list = entities['augmented_blink_entity_candidates_list']
        entity_candidates_list = entities[args.entity_candidate_name]

        processed_entity_names = []
        predict_entity_names = []
        for (
            entity_name,
            entity_candidates,
            # multi_choice_prompt_result,
        ) in zip(
            entity_names,
            entity_candidates_list,
            # multi_choice_prompt_results,
        ):
            # the entity name may have different format (disambiguation, etc)
            processed_entity_name = args.wikipedia.preprocess_ent_name(preprocess(entity_name))
            processed_entity_name = preprocess(processed_entity_name).lower()
            entity_candidates = [preprocess(tmp).lower() for tmp in entity_candidates]
            # predict_entity_name = process_multi_choice_prompt(multi_choice_prompt_result, entity_candidates)
            predict_entity_name = ''

            processed_entity_names.append(processed_entity_name)
            predict_entity_names.append(predict_entity_name)

            if processed_entity_name != '':
                num_gt_instance += 1
                # print(f'processed_entity_name: {processed_entity_name}')
                # print(f'entity_candidates: {entity_candidates}')
                # import sys
                # sys.exit()
                if processed_entity_name in entity_candidates:
                    if processed_entity_name in entity_candidates[:1]:
                        top_1 += 1
                    if processed_entity_name in entity_candidates[:3]:
                        top_3 += 1
                    if processed_entity_name in entity_candidates[:5]:
                        top_5 += 1
                    if processed_entity_name in entity_candidates[:10]:
                        top_10 += 1 
                else:
                    print(f'processed_entity_name: {processed_entity_name}')
                    print(f'entity_candidates: {entity_candidates}')
                    # pass

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

    # gold_recall = dev_by_zero(gold_true_positve, num_gt_instance)

    top_1 = dev_by_zero(top_1, num_gt_instance)
    top_3 = dev_by_zero(top_3, num_gt_instance)
    top_5 = dev_by_zero(top_5, num_gt_instance)
    top_10 = dev_by_zero(top_10, num_gt_instance)

    out_dict = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        # 'gold_recall': gold_recall,
        'recall@1': top_1,
        'recall@3': top_3,
        'recall@5': top_5,
        'recall@10': top_10,
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
    wikipedia = Wikipedia(args.base_url, args.wiki_version)
    args.wikipedia = wikipedia
    input_file = args.input_file
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)
    out_dict = evaluate_ed_chatgpt_multi_choice(args, doc_name2instance)
    print(out_dict)

if __name__ == '__main__':
    main()