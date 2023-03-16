

def dev_by_zero(a, b):
        if b == 0:
            return 0.0
        else:
            return a / b
        

def evaluate_and_analysis(doc_name2instance):
    
    # doc_name2inference: predict
    # doc_name2instance: gt

    for doc_name, instance in doc_name2instance.items():

    num_pred_instance = 0
    num_gt_instance = 0
    num_true_positive = 0

    for doc_name in doc_name2instance:
        gt_entities = doc_name2instance[doc_name]['entities']
        position2entity_name = dict()
        for entity_start, entity_end, entity_name in zip(
                gt_entities['starts'], gt_entities['ends'], gt_entities['entity_names'],
        ):
            if entity_name != '':
                num_gt_instance += 1
            position2entity_name[(entity_start, entity_end)] = entity_name

        if doc_name not in doc_name2inference:
            continue
        else:
            pred_entities = doc_name2inference[doc_name]['entities']
            for entity_start, entity_end, entity_name, entity_ner_label in zip(
                    pred_entities['starts'], pred_entities['ends'], pred_entities['entity_names'],
                    pred_entities.get('entity_ner_labels', 'NULL')
            ):
                num_pred_instance += 1
                if (entity_start, entity_end) in position2entity_name and \
                        position2entity_name[(entity_start, entity_end)] == entity_name:
                    num_true_positive += 1
                    correct_entity_disambiguation_by_ner[entity_ner_label] += 1

                if (entity_start, entity_end) in position2entity_name and \
                        position2entity_name[(entity_start, entity_end)] != entity_name:
                    incorrect_entity_disambiguation_by_ner[entity_ner_label] += 1
                    if position2entity_name[(entity_start, entity_end)] == '':
                        num_pred_instance -= 1

            # compute gold_recall assuming all the GT in candidate entities can correctly selected
            if 'entity_candidates' in pred_entities:
                for entity_start, entity_end, entity_name, entity_ner_label, entity_candidate in zip(
                        pred_entities['starts'], pred_entities['ends'], pred_entities['entity_names'],
                        pred_entities.get('entity_ner_labels', 'NULL'), pred_entities['entity_candidates']
                ):
                    if (entity_start, entity_end) in position2entity_name and \
                            position2entity_name[(entity_start, entity_end)] in entity_candidate:
                        num_gold_true_positive += 1

    precision = dev_by_zero(num_true_positive, num_pred_instance)
    recall = dev_by_zero(num_true_positive, num_gt_instance)
    f1 = dev_by_zero(2 * precision * recall, precision + recall)

    gold_recall = dev_by_zero(num_gold_true_positive, num_gt_instance)
    print(f'num_true_positive: {num_true_positive}; num_pred_instance: {num_pred_instance}; num_gt_instance: {num_gt_instance}')
    print(f'incorrect_entity_disambiguation_by_ner: {incorrect_entity_disambiguation_by_ner}')
    print(f'correct_entity_disambiguation_by_ner: {correct_entity_disambiguation_by_ner}')
    print(f'precision: {precision}, recall: {recall}, f1: {f1}')
    print(f'gold_recall: {gold_recall}')

    out_dict = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gold_recall': gold_recall,
        'num_true_positive': num_true_positive,
        'num_pred_instance': num_pred_instance,
        'num_gt_instance': num_gt_instance,
        'incorrect_entity_disambiguation_by_ner': incorrect_entity_disambiguation_by_ner,
        'correct_entity_disambiguation_by_ner': correct_entity_disambiguation_by_ner,
    }
    return out_dict