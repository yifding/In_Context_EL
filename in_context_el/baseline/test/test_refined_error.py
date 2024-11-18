import os
import json

def test_refined(sentence, spans):
    from refined.inference.processor import Refined
    from refined.data_types.base_types import Span
    from in_context_el.baseline.refined.generate import refined4ed_el
    model_name = "aida_model"
    entity_set = "wikipedia"
    refined = Refined.from_pretrained(
        model_name=model_name,
        entity_set=entity_set,
        use_precomputed_descriptions=True,
    )

    print('test RefinED for entity disambiguation: \n')
    output = refined4ed_el(sentence, spans, refined, el=False, return_ori=False)
    print(json.dumps(output, indent=4))

    print('test RefinED for entity linking: \n')
    output = refined4ed_el(sentence, spans, refined, el=True, return_ori=False)
    print(json.dumps(output, indent=4))


# '''
sentence = "England won the FIFA World Cup in 1966."
spans = [
    (0, 7),
    (16, 30),
]
test_refined(sentence, spans)
# '''

# from transformers import (
#     AutoTokenizer,
#     PreTrainedTokenizerFast,
# )

# data_dir = '/home/yding4/.cache/refined'
# transformer_name = 'roberta-base'
# tokenizer = AutoTokenizer.from_pretrained(
#     os.path.join(data_dir, transformer_name),
#     use_fast=True,
#     # add_special_tokens=False,
#     # add_prefix_space=False,
# )