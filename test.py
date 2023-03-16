import os
import json
import openai
openai.api_key = "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"
import torch
import requests
import argparse
import collections
from tqdm import tqdm

from REL.wikipedia import Wikipedia
from REL.wikipedia_yago_freq import WikipediaYagoFreq
from transformers import AutoTokenizer, AutoModelForSequenceClassification
API_URL = "http://localhost:5556"

# # load Wikipedia
# base_url ="/nfs/yding4/REL/data/"
# wiki_version="wiki_2014"
# wikipedia = Wikipedia(base_url, wiki_version)
# wikipedia_yago_freq = WikipediaYagoFreq(base_url, wiki_version, wikipedia)
# wikipedia_yago_freq.extract_entity_description()

# # load NLI model
# hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
# model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)


def load_tsv(file, key='', mode='char'):
    def process_token_2_char_4_doc_name2instance(token_doc_name2instance):
        char_doc_name2instance = dict()
        for doc_name, instance in token_doc_name2instance.items():
            starts = []
            ends = []
            entity_mentions = []
            entity_names = []
            assert doc_name == instance['doc_name']
            tokens = instance['tokens']
            sentence = ' '.join(tokens)
            token_entities = instance['entities']

            for token_start, token_end, token_entity_mention, token_entity_name in zip(
                    token_entities['starts'], token_entities['ends'], token_entities['entity_mentions'],
                    token_entities['entity_names']
            ):
                if not 0 <= token_start <= token_end < len(tokens):
                    print(instance)

                assert 0 <= token_start <= token_end < len(tokens)
                # **YD** sentence[char_start: char_end] == mention
                # **YD** ' '.join(tokens[token_start: token_end+1]) == mention ## ignoring the ',', '.' without space cases
                if token_start == 0:
                    start = 0
                else:
                    start = len(' '.join(tokens[:token_start])) + 1
                end = len(' '.join(tokens[:token_end + 1]))
                entity_mention = sentence[start: end]

                starts.append(start)
                ends.append(end)
                entity_mentions.append(entity_mention)
                entity_names.append(token_entity_name)

            char_doc_name2instance[doc_name] = {
                'doc_name': doc_name,
                # 'tokens': tokens,
                'sentence': sentence,
                'entities': {
                    "starts": starts,
                    "ends": ends,
                    "entity_mentions": entity_mentions,
                    "entity_names": entity_names,
                }
            }
        return char_doc_name2instance

    def generate_instance(
        doc_name,
        tokens,
        ner_tags,
        entity_mentions,
        entity_names,
        entity_wikipedia_ids,
    ):
        assert len(tokens) == len(ner_tags) == len(entity_mentions) \
               == len(entity_names) == len(entity_wikipedia_ids)

        instance_starts = []
        instance_ends = []
        instance_entity_mentions = []
        instance_entity_names = []
        instance_entity_wikipedia_ids = []

        tmp_start = -1
        for index, (ner_tag, entity_mention, entity_name, entity_wikipedia_id) in enumerate(
                zip(ner_tags, entity_mentions, entity_names, entity_wikipedia_ids)
        ):

            # judge whether current token is the last token of an entity, if so, generate an entity.
            if ner_tag == 'O':
                continue
            else:
                if ner_tag.startswith('B'):
                    # if the index hits the last one or next ner_tag is not 'I',
                    if index == len(tokens) - 1 or ner_tags[index + 1].startswith('B') or ner_tags[index + 1] == 'O':
                        instance_starts.append(index)
                        instance_ends.append(index)
                        # the end_index select the last token of a entity to allow simple index
                        # location for a transformer tokenizer
                        instance_entity_mentions.append(entity_mention)
                        instance_entity_names.append(entity_name)
                        instance_entity_wikipedia_ids.append(entity_wikipedia_id)
                        tmp_start = -1
                    else:
                        tmp_start = index
                else:
                    assert ner_tag.startswith('I')
                    if index == len(tokens) - 1 or ner_tags[index + 1].startswith('B') or ner_tags[index + 1] == 'O':
                        instance_starts.append(tmp_start)
                        instance_ends.append(index)
                        # the end_index select the last token of a entity to allow simple index
                        # location for a transformer tokenizer
                        instance_entity_mentions.append(entity_mention)
                        instance_entity_names.append(entity_name)
                        instance_entity_wikipedia_ids.append(entity_wikipedia_id)

                        assert tmp_start != -1
                        tmp_start = -1

        instance = {
            'doc_name': doc_name,
            'tokens': tokens,
            'entities': {
                "starts": instance_starts,
                "ends": instance_ends,
                "entity_mentions": instance_entity_mentions,
                "entity_names": instance_entity_names,
                "entity_wikipedia_ids": instance_entity_wikipedia_ids,
            }
        }
        return instance

    doc_name2dataset = dict()
    doc_name = ''
    tokens = []
    ner_tags = []
    entity_mentions = []
    entity_names = []
    entity_wikipedia_ids = []

    assert all(token != ' ' for token in tokens)

    with open(file) as reader:
        for line in reader:
            if line.startswith('-DOCSTART-'):
                if tokens:
                    assert doc_name != ''
                    assert doc_name not in doc_name2dataset

                    instance = generate_instance(
                        doc_name,
                        tokens,
                        ner_tags,
                        entity_mentions,
                        entity_names,
                        entity_wikipedia_ids,
                    )
                    if key in doc_name:
                        doc_name2dataset[doc_name] = instance

                    tokens = []
                    ner_tags = []
                    entity_mentions = []
                    entity_names = []
                    entity_wikipedia_ids = []

                assert line.startswith('-DOCSTART- (')
                tmp_start_index = len('-DOCSTART- (')
                if line.endswith(')\n'):
                    tmp_end_index = len(')\n')
                else:
                    tmp_end_index = len('\n')
                doc_name = line[tmp_start_index: -tmp_end_index]
                assert doc_name != ''

            elif line == '' or line == '\n':
                continue

            else:
                parts = line.rstrip('\n').split("\t")
                # len(parts) = [1, 4, 6, 7]
                # 1: single symbol
                # 4: ['Tim', 'B', "Tim O'Gorman", '--NME--'] or ['David', 'B', 'David', 'David_Beckham']
                # 6: ['House', 'B', 'House of Commons', 'House_of_Commons', 'http://en.wikipedia.org/wiki/House_of_Commons', '216091']
                # 7: ['German', 'B', 'German', 'Germany', 'http://en.wikipedia.org/wiki/Germany', '11867', '/m/0345h']
                assert len(parts) in [1, 4, 6, 7]

                # Gets out of unicode storing in the entity names
                # example: if s = "\u0027", in python, it will be represented as "\\u0027" and not recognized as an
                # unicode, should do .encode().decode("unicode-escape") to output "\'"
                if len(parts) >= 4:
                    parts[3] = parts[3].encode().decode("unicode-escape")

                # 1. add tokens
                # the extra space may destroy the position of token when creating sentences
                # tokens.append(parts[0].replace(' ', '_'))
                tokens.append(parts[0])

                # 2. add ner_tags
                if len(parts) == 1:
                    ner_tags.append('O')
                else:
                    ner_tags.append(parts[1])

                # 3. add entity_names
                if len(parts) == 1:
                    entity_mentions.append('')
                    entity_names.append('')
                else:
                    entity_mentions.append(parts[2])
                    if parts[3] == '--NME--':
                        entity_names.append('')
                    else:
                        entity_names.append(parts[3])

                # 4. add entity_wikiid if possible (only aida dataset has wikiid)
                if len(parts) >= 6 and int(parts[5]) > 0:
                    wikipedia_id = int(parts[5])
                    entity_wikipedia_ids.append(wikipedia_id)
                else:
                    entity_wikipedia_ids.append(-1)

    if tokens:
        assert doc_name != ''
        assert doc_name not in doc_name2dataset

        instance = generate_instance(
            doc_name,
            tokens,
            ner_tags,
            entity_mentions,
            entity_names,
            entity_wikipedia_ids,
        )
        if key in doc_name:
            doc_name2dataset[doc_name] = instance

    if mode == 'token':
        return doc_name2dataset
    else:
        assert mode == 'char', 'MODE(parameter) only supports "token" and "char"'
        return process_token_2_char_4_doc_name2instance(doc_name2dataset)


def nli_score(premise:str, hypothesis: str, model, tokenizer):
    tokenized_input_seq_pair = tokenizer.encode_plus(
        premise, 
        hypothesis,
        max_length=512,
        return_token_type_ids=True, 
        truncation=True
    )

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()[0]
    return predicted_probability


# 1. load dataset
input_file = '/nfs/yding4/EL_project/dataset/KORE50/AIDA.tsv'
doc_name2instance = load_tsv(input_file)

# 2. Given mention, load entity candidates and corresponding entity description
# lazy idea: utilize REL local API to obtain entity candidates
for key, value in doc_name2instance.items():
    sentence = value['sentence']
    el_result = requests.post(API_URL, json={
        "text": sentence,
        "spans": []
    }).json()
    # is a list of things, each item is a mention-entity prediction pair.
    print(sentence)
    for m_e_pre in el_result:
        entity_candidates = m_e_pre[7]
        mention = m_e_pre[3]
        prompt = sentence + " \n What does " + mention + "in this sentence referring to?"
        openai_output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
            {"role": "system", "content": prompt},
            ]
        )
        complete_output = openai_output["choices"][0]["message"]['content']
        print(f'complete_output: {complete_output}')
        prompt_match_score = []
    #     for entity_candidate in entity_candidates:
    #         # find its corresponding wikipedia description
    #         id = wikipedia.ent_wiki_id_from_name(entity_candidate)
    #         if id in wikipedia_yago_freq.entity_id2description:
    #             entity_description = wikipedia_yago_freq.entity_id2description[id][:200]
    #             entity_description = entity_description.split('\n\n')[1][:200]
    #         else:
    #             entity_description = entity_candidate
    #         score = nli_score(complete_output, entity_description, model, tokenizer)
    #         print(f'entity_candidate: {entity_candidate}; entity_description: {entity_description}; score: {score}')
    #         break
    #     break
    # break


# 4. use NLI to align chatgpt prediction with entity description

