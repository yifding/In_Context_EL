import os
import jsonlines
from tqdm import tqdm

input_dir = "/nfs/yding4/In_Context_EL/data/ed/fine_tuning/ed_aida/openai"
output_dir = "/nfs/yding4/In_Context_EL/data/ed/fine_tuning/ed_aida/replicate"
os.makedirs(output_dir, exist_ok=True)
for tmp_input_file in ['aida_testa_gpt.jsonl', 'aida_train_gpt.jsonl']:
    input_file = os.path.join(input_dir, tmp_input_file)
    output_file = os.path.join(output_dir, tmp_input_file)
    with jsonlines.open(input_file) as reader:
        records = [record for record in reader]
    new_records = []
    for record in tqdm(records):
        messages = record['messages']
        assert len(messages) == 2
        prompt = messages[0]['content']
        completion = messages[1]['content']
        new_record = {'prompt': prompt, 'completion': completion}
        new_records.append(new_record)
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(new_records)