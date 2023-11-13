import os
import openai
import jsonlines
from tqdm import tqdm, trange
from in_context_el.openai_key import OPENAI_API_KEY
from in_context_el.openai_function import openai_chatgpt, openai_completion
openai.api_key = OPENAI_API_KEY
from timeout_decorator import timeout


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


input_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/10_25_2023/blink_candidates/bbn_test.json'
output_file = '/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/10_25_2023/blink_candidates/bbn_test_one_step_results.json'


if os.path.isfile(output_file):
    with jsonlines.open(output_file) as reader:
        records = [record for record in reader]
else:
    with jsonlines.open(input_file) as reader:
        records = [record for record in reader]

# index = 0
# pbar = tqdm(total=len(records))
# while index < len(records):
#     record = records[index]
#     if 'prompt_answer' in record:
#         index += 1
#         print(f'{index} has been skipped')
#         pbar.update(1)
#         continue

max_retries = 5

for i, record in enumerate(tqdm(records, total=len(records))):
    if 'prompt_answer' not in record:
        sentence = record['left_context_text'] + ' ' + record['word'] + ' ' + record['right_context_text']
        prompt_answer = dict()
        FLAG = True

        retry = 0
        for entity_type in reverse_train_types:
            prompt = f'Please answer the following question with yes or no.' \
                     f'{sentence}\n' \
                     f'is/was {record["word"]} a {entity_type} ?'
            while retry < max_retries:
                    try:
                        complete_output = openai_chatgpt(prompt, model='gpt-3.5-turbo')
                        prompt_answer[entity_type] = complete_output
                        break

                    except:
                        retry += 1
        records[i]['prompt_answer'] = prompt_answer
        with jsonlines.open(output_file, 'w') as writer:
            writer.write_all(records)

        # for entity_type in reverse_train_types:
        #     prompt = f'Please answer the following question with yes or no.' \
        #              f'{sentence}\n' \
        #              f'is/was {record["word"]} a {entity_type} ?'
        #     try:
        #         complete_output = openai_chatgpt(prompt, model='gpt-3.5-turbo')
        #
        #     except:
        #         FLAG = False
        #         complete_output = ''
        #     else:
        #         pass
        #     prompt_answer[entity_type] = complete_output
        # if FLAG:
        #     records[i]['prompt_answer'] = prompt_answer
        #     # with jsonlines.open(output_file, 'w') as writer:
        #     #     writer.write_all(records)
        #     # pbar.update(1)
        # else:
        #     i -= 1


