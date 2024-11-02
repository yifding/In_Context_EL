import faiss
import jsonlines
import numpy as np
from tqdm import tqdm

# from REL.wikipedia import Wikipedia

from transformers import AutoTokenizer, AutoModel
import os
import json


input_json_file = '/nfs/yding4/FirstSent/RUN_FILES/process/test.jsonl'
sentence2emb_file = '/nfs/yding4/FirstSent/RUN_FILES/process/sentence2emb.npy'

device = 'cuda:1'
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
model = model.to(device).eval()

# base_url = '/nfs/yding4/REL/data/'
# wiki_version = 'wiki_2014'
# load wikipedia
# wikipedia = Wikipedia(base_url, wiki_version)
k = 10

'''
index2name = dict()
name2index = dict()

with jsonlines.open(input_json_file) as reader:
    for index, instance in enumerate(tqdm(reader)):
        ent_name = instance['ent_name']
        index2name[index] = ent_name
        name2index[ent_name] = index

print('start to load sentence embedding!')
sentence2emb = np.load(sentence2emb_file)
# sentence2emb = np.load('sentence2emb_2.npy')
print('load sentence embedding is done!')


# create faiss index
faiss_index = faiss.IndexFlatL2(len(sentence2emb[0]))
faiss_index.add(sentence2emb)
print(faiss_index.ntotal)


entity_list_1 = ['David_Beckham', 'Victoria_Beckham', 'U.S._Open_(golf)', 'Tiger_Woods', 'google', 'Microsoft', 'Apple Inc.', 'China national football team', 'Chicago Bulls', 'New England Patriots', 'Washington Capitals', 'Michael Jordan', 'Donald Knuth']
entity_list_2 = ['South Bend, Indiana', 'Chongqing', 'Chicago', 'George W. Bush', 'Naruto', 'University of Notre Dame', 'Massachusetts Institute of Technology', 'Stanford University', 'Harvard University']
entity_list_3 = ['Beijing University of Posts and Telecommunications', 'University of Science and Technology of China', 'Tsinghua University', 'Xi Jinping', 'Jiang Zemin']
entity_list = entity_list_1 + entity_list_2 + entity_list_3
    # ent_name = wikipedia.preprocess_ent_name(entity)

for entity in entity_list:
    ent_name = entity
    if ent_name not in name2index:
        continue

    ent_index = name2index[ent_name]
    print(f'ent_name: {ent_name}; ent_index: {ent_index}')
    source_ent_emb = sentence2emb[ent_index: ent_index + 1, :]
    D, I = faiss_index.search(source_ent_emb, k)
    neighbor_ent_names = [index2name[int(i)] for i in np.nditer(I) if i != ent_index]
    print(neighbor_ent_names)
    print('\n')
'''




# 1. load json file
input_file = '/nfs/yding4/In_Context_EL/RUN_FILES/10_13_2024/augmented_blink_candidates/KORE50.json'
with open(input_file) as reader:
        doc_name2instance = json.load(reader)

for doc_name, instance in tqdm(doc_name2instance.items()):
    sentence = instance['sentence']
    entities = instance['entities']
    entity_candidates_list = []
    for (
        entity_name,
        prompt_result,

    ) in zip(
        entities['entity_names'],
        entities['prompt_results'],
    ):
        prompt_result = 'Johnny Cash, the legendary American singer-songwriter and musician known for his deep, distinctive voice and his influence on country, rock, and other music genres.'
        prompt_ids = tokenizer(
            prompt_result,
            truncation=True,
            max_length=50,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        prompt_emb = model(**prompt_ids)[0][0, 0, :].cpu().detach().numpy()
        D, I = faiss_index.search(prompt_emb.reshape(1,-1), k)
        neighbor_ent_names = [index2name[int(i)] for i in np.nditer(I)]
        print('entity_name', entity_name)
        print('neighbor_ent_names', neighbor_ent_names)
        print('\n')
        break
    break

