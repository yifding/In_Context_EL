import os
import json
from tqdm import tqdm


input_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/ED_standard_datasets'
output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/baseline/LLM4ED/ED_standard_datasets/prediction/'


datasets = ['clueweb']

for dataset in datasets:
    print(dataset)
    input_file = os.path.join(input_dir, dataset + '.json')
    output_file = os.path.join(output_dir, dataset + '.json')
    if os.path.isfile(output_file):
        input_file = output_file

    with open(input_file) as reader:
        doc_name2instance = json.load(reader)

    for doc_index, (doc_name, instance) in tqdm(enumerate(doc_name2instance.items())):
        # print(instance.keys())
        if 'out_dicts' in instance:
            continue
        else:
            print(doc_index, 'yyaya')
            break
    