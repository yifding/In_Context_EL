import os
import json
import torch
import argparse
from tqdm import tqdm
from refined.inference.processor import Refined
from refined.data_types.base_types import Span

# https://github.com/amazon-science/ReFinED/blob/main/replicate_results.py

def refined4ed_el(sentence, spans, refined, el=False, return_ori=False):
    refined_spans = []
    for span in spans:
        assert len(span) == 2
        start, end = span
        refined_span = Span(text=sentence[start: end], start=start, ln=end-start)
        refined_spans.append(refined_span)

    if el:
        pred_spans = refined.process_text(
                sentence, 
        )
    else:
        pred_spans = refined.process_text(
                sentence, 
                spans = refined_spans,
        )

    if return_ori:
        return [] if pred_spans is None else pred_spans
    
    starts = []
    ends = []
    entity_mentions = []
    entity_names = []

    if pred_spans is not None:
        for pred_span in pred_spans:
            start = pred_span.start
            mention = str(pred_span.text)
            end = start + len(mention)
            entity = pred_span.predicted_entity.wikipedia_entity_title
            if entity == '' or entity is None:
                entity = '' 
            starts.append(start)
            ends.append(end)
            entity_mentions.append(mention)
            entity_names.append(entity)

    pred_entities = {
        'starts': starts,
        'ends': ends,
        'entity_mentions': entity_mentions,
        'entity_names': entity_names,
    }
    return pred_entities


def parse_args():
    parser = argparse.ArgumentParser(
        description='argument for use RefinED for ED/EL',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_dir",
        help="the processed dataset file",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/ED",
        type=str,
    )
    parser.add_argument(
        "--datasets",
        help="the processed dataset file",
        # required=True,
        default="['aida_testb','msnbc','aquaint','ace2004','clueweb','wikipedia','KORE50','oke_2015','oke_2016','Reuters-128','RSS-500']",
        type=eval,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        # required=True,
        default="/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/baseline/refined/ED/prediction",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        help="model parameter",
        # required=True,
        choices=["aida_model", "wikipedia_model"],
        default="wikipedia_model",
        type=str,
    )
    parser.add_argument(
        "--entity_set",
        help="entity set for the model",
        # required=True,
        choices=["wikipedia"],
        default="wikipedia",
        type=str,
    )
    parser.add_argument(
        "--el",
        help="true for predicting entity linking, false for entity disambiguation",
        action='store_true',
    )
    parser.add_argument(
        "--device",
        default='cpu',
        help="device for refined model",
    )



    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    assert os.path.isdir(args.input_dir)
    return args


def predict_ed_el():
    args = parse_args()
    refined = Refined.from_pretrained(
        model_name=args.model_name,
        entity_set=args.entity_set,
        device=torch.device(args.device)
    )

    for dataset in args.datasets:
        print(f'dataset: {dataset}')
        input_file = os.path.join(args.input_dir, dataset + '.json')
        output_file = os.path.join(args.output_dir, dataset + '.json')
            
        with open(input_file) as reader:
            doc_name2instance = json.load(reader)

        for doc_name, instance in tqdm(doc_name2instance.items()):
            sentence = instance['sentence']
            entities = instance['entities']

            span_list = []
            for (
                start,
                end,
            ) in zip(
                entities['starts'],
                entities['ends'],
            ):
                span = (start, end)
                span_list.append(span)

            pred_entities = refined4ed_el(
                    sentence, 
                    span_list,
                    refined,
                    el=args.el,
                    return_ori=False,
            )

            doc_name2instance[doc_name]['pred_entities'] = pred_entities

            with open(output_file, 'w') as writer:
                json.dump(doc_name2instance, writer, indent=4)
    

def test_refined_ed():
    
    sentence = "England won the FIFA World Cup in 1966."
    model_name = "aida_model"
    entity_set = "wikipedia"
    refined = Refined.from_pretrained(
        model_name=model_name,
        entity_set=entity_set,
        use_precomputed_descriptions=True,
    )
    spans = [
        (0, 7),
        (8, 11),
        (16, 30),
    ]
    ed_out = refined4ed_el(sentence, spans, refined, el=False, return_ori=True)
    # spans = refined.process_text("England won the FIFA World Cup in 1966.")
    print(ed_out)
    
    el_out = refined4ed_el(sentence, spans, refined, el=True, return_ori=True)
    print(el_out)


if __name__ == '__main__':
    # main()
    # test_refined_ed()
    predict_ed_el()