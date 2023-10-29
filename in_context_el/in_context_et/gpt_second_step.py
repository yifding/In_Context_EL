import os
import jsonlines
import nltk
from nltk.tokenize import sent_tokenize
from in_context_el.openai_key import OPENAI_API_KEY
from in_context_el.openai_function import openai_chatgpt, openai_completion
from tqdm import tqdm
import openai
openai.api_key = OPENAI_API_KEY


def read_type_hierarchy(old_hier):
    train_types = {}
    reverse_train_types = {}
    with open(old_hier) as f:
        for line in f:
            if (len(line.strip().split(':')) < 2):
                continue
            tmp = line.strip().split(':')
            # if tmp[0] not in train_type_count:
            #     continue
            train_types[tmp[0]] = tmp[1]
            reverse_train_types[tmp[1]] = tmp[0].split('/')[1:]
    print('\ntrain_types:')
    print(train_types)
    print('\nreverse_train_types:')
    print(reverse_train_types)
    return train_types, reverse_train_types


old_hier = '/afs/crc.nd.edu/user/y/yding4/ET_project/dataset/bbn/bbn_types.txt'
train_types, reverse_train_types = read_type_hierarchy(old_hier)

# input_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/10_25_2023/blink_candidates/sample_10_bbn_test.json'
# output_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/10_25_2023/blink_candidates/sample_10_results.json'
input_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/10_25_2023/blink_candidates/bbn_test.json'
output_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/10_25_2023/blink_candidates/bbn_test_results.json'

if os.path.isfile(output_file):
    with jsonlines.open(output_file) as reader:
        records = [record for record in reader]
else:
    with jsonlines.open(input_file) as reader:
        records = [record for record in reader]


soft_sentence_limit = 150
for index, record in enumerate(tqdm(records)):
    if 'prompt_answer' in record:
        continue
    entity_candidates = record['entity_candidates']
    entity_candidates_descriptions = record['entity_candidates_descriptions']
    entity_candidate = entity_candidates[0]
    entity_candidates_description = entity_candidates_descriptions[0]
    tokenzied_sentences = sent_tokenize(entity_candidates_description)
    tmp_descriptions = ''
    tmp_sent_index = 0
    while len(tmp_descriptions) < soft_sentence_limit and tmp_sent_index < len(tokenzied_sentences):
        if tmp_descriptions != '':
            tmp_descriptions += ' '
        tmp_descriptions += tokenzied_sentences[tmp_sent_index]
        tmp_sent_index += 1

    prompt_answer = dict()
    for entity_type in reverse_train_types:
        prompt = f'Please answer the following question with yes or no.' \
                 f'{tmp_descriptions}\n' \
                 f'is/was {entity_candidate} a {entity_type} ?'
        complete_output = openai_chatgpt(prompt, model='gpt-3.5-turbo-16k')
        prompt_answer[entity_type] = complete_output
    records[index]['prompt_answer'] = prompt_answer

    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(records)


