import os
import json
import requests
import argparse
import collections
import urllib
import rdflib
from tqdm import tqdm
from collections import defaultdict


def load_tsv(file='/nfs/yding4/EL_project/dataset/KORE50/AIDA.tsv', key='', mode='char'):
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


def load_ttl_oke_2015(
    file='/nfs/yding4/EL_project/dataset/oke-challenge/evaluation-data/task1/evaluation-dataset-task1.ttl',
):
    def process_sen_char(s):
        assert 'sentence-' in s
        first_parts = s.split('sentence-')
        assert len(first_parts) == 2
        assert '#char=' in first_parts[1]
        second_parts = first_parts[1].split('#char=')
        assert len(second_parts) == 2
        assert ',' in second_parts[1]
        sentence_index = int(second_parts[0])
        third_parts = second_parts[1].split(',')
        assert len(third_parts) == 2
        char_start, char_end = int(third_parts[0]), int(third_parts[1])
        return char_start, char_end, sentence_index

    g = rdflib.Graph()
    g.parse(file, format='ttl')

    module_list = [
        'label', 'anchorOf', 'beginIndex', 'isString', 'sameAs', 'endIndex', 'taIdentRef', 'referenceContext', 'type',
    ]

    # 1. isString: extracts sentence (identified by the sentence number)
    # 2. taIdentRef: extracts mentions and labelled temporary annotations
    # 3. sameAs: bring temporary annotations to dataset-base if database has corresponding entities
    sentence_index2sentence = dict()
    sent_char_index2tmp_entity = dict()
    tmp_entity2entity = dict()

    for node_index, node in enumerate(g):
        parts = node[1].split('#')
        assert len(parts) == 2
        assert parts[1] in module_list

        if parts[1] == 'anchorOf':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            tmp_str = str(node[2]).rstrip()
            if (char_end - char_start) != len(tmp_str):
                # only one data error: 'Basel, Switzerland'
                tmp_str = tmp_str.split(',')[0]
            assert (char_end - char_start) == len(tmp_str)

        elif parts[1] == 'taIdentRef':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert str(node[2]).count('sentence-') == 1
            tmp_entity = str(node[2]).split('sentence-')[1]
            assert (sentence_index, char_start, char_end) not in sent_char_index2tmp_entity or \
                   sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] in ['Man_4', 'His_4']
            sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] = tmp_entity

        elif parts[1] == 'isString':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert sentence_index not in sentence_index2sentence
            sentence_index2sentence[sentence_index] = str(node[2])

        elif parts[1] == 'sameAs':
            assert str(node[0]).count('sentence-') == 1
            mention = str(node[0]).split('sentence-')[1]

            entity = str(node[2]).split('/')[-1]
            if mention in tmp_entity2entity:
                assert entity == tmp_entity2entity[mention]
            tmp_entity2entity[mention] = entity
            # print(mention, entity)
            # if not str(node[2]).startswith('http://dbpedia.org/resource/'):
            #     print(node)
        else:
            if parts[1] == 'label':
                # 'label' is not useful
                tmp_split_str = str(node[0]).split('sentence-')[1]
                tmp_str = str(node[2])
                assert tmp_split_str == tmp_str.replace(' ', '_')

    num_in = 0
    num_out = 0

    sorted_key = sorted(sent_char_index2tmp_entity.keys(), key=lambda x: (x[0], x[1], x[2]))
    doc_name2instance = dict()
    for (tmp_sent_index, char_start, char_end) in sorted_key:
        sentence = sentence_index2sentence[tmp_sent_index]
        if str(tmp_sent_index) not in doc_name2instance:
            doc_name2instance[str(tmp_sent_index)] = {
                'sentence': sentence,
                'entities': {
                    'starts': [],
                    'ends': [],
                    'entity_mentions': [],
                    'entity_names': [],
                }
            }
        tmp_entity = sent_char_index2tmp_entity[(tmp_sent_index, char_start, char_end)]
        processed_tmp_entity = tmp_entity.replace(' ', '_')
        if processed_tmp_entity in tmp_entity2entity:
            num_in += 1
            entity = tmp_entity2entity[processed_tmp_entity]
            # assert (char_end - char_start) == len(tmp_str)
            mention = sentence[char_start: char_end]
            doc_name2instance[str(tmp_sent_index)]['entities']['starts'].append(char_start)
            doc_name2instance[str(tmp_sent_index)]['entities']['ends'].append(char_end)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_mentions'].append(mention)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_names'].append(entity)

        else:
            num_out += 1
    print(f'num_in_kb: {num_in}; num_out_kb: {num_out}; len(tmp_entity2entity): {len(tmp_entity2entity)}')
    # print(json.dumps(doc_name2instance, indent=4))
    return doc_name2instance


def load_unseen_mentions(file='/nfs/yding4/EL_project/dataset/unseen_mentions/test.json'):
    doc_name2instance = dict()
    with open(file) as reader:
        for index, line in enumerate(reader):
            d = json.loads(line)
            # doc_name = str(d['docId'])
            doc_name = str(index)
            mention = ' '.join(d['mention_as_list'])
            entity = d['y_title']
            sentence = d['left_context_text'] + ' ' + mention + ' ' + d['right_context_text']
            start = len( d['left_context_text']) + 1
            end = start + len(mention)

            doc_name2instance[doc_name] = {
                'sentence': sentence,
                'entities': {
                    'starts': [start],
                    'ends': [end],
                    'entity_mentions': [mention],
                    'entity_names': [entity],
                }
            }
    
    return doc_name2instance


def load_ttl_oke_2016(
    file='/nfs/yding4/EL_project/dataset/oke-challenge-2016/evaluation-data/task1/evaluation-dataset-task1.ttl',
):
    def process_sen_char(s):
        assert 'sentence-' in s
        first_parts = s.split('sentence-')
        assert len(first_parts) == 2
        assert '#char=' in first_parts[1]
        second_parts = first_parts[1].split('#char=')
        assert len(second_parts) == 2
        assert ',' in second_parts[1]
        sentence_index = int(second_parts[0])
        third_parts = second_parts[1].split(',')
        assert len(third_parts) == 2
        char_start, char_end = int(third_parts[0]), int(third_parts[1])
        return char_start, char_end, sentence_index

    g = rdflib.Graph()
    g.parse(file, format='ttl')

    module_list = [
        'label', 'anchorOf', 'beginIndex', 'isString', 'sameAs', 'endIndex', 'taIdentRef', 'referenceContext', 'type',
    ]

    # 1. isString: extracts sentence (identified by the sentence number)
    # 2. taIdentRef: extracts mentions and labelled temporary annotations
    # 3. sameAs: bring temporary annotations to dataset-base if database has corresponding entities
    sentence_index2sentence = dict()
    sent_char_index2tmp_entity = dict()
    tmp_entity2entity = dict()

    for node_index, node in enumerate(g):
        parts = node[1].split('#')
        assert len(parts) == 2
        assert parts[1] in module_list

        if parts[1] == 'anchorOf':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            tmp_str = str(node[2]).rstrip()
            if (char_end - char_start) != len(tmp_str):
                # only one data error: 'Basel, Switzerland'
                tmp_str = tmp_str.split(',')[0]
            assert (char_end - char_start) == len(tmp_str)

        elif parts[1] == 'taIdentRef':
            # print(node)
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert str(node[2]).count('task-1/') == 1
            tmp_entity = str(node[2]).split('task-1/')[1]
            assert (sentence_index, char_start, char_end) not in sent_char_index2tmp_entity or \
                   sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] in ['Man_4', 'His_4']
            sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] = tmp_entity

        elif parts[1] == 'isString':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert sentence_index not in sentence_index2sentence
            sentence_index2sentence[sentence_index] = str(node[2])

        elif parts[1] == 'sameAs':
            # print(node)
            assert str(node[0]).count('task-1/') == 1
            mention = str(node[0]).split('task-1/')[1]

            entity = str(node[2]).split('/')[-1]
            if mention in tmp_entity2entity:
                assert entity == tmp_entity2entity[mention]
            tmp_entity2entity[mention] = entity
            # print(mention, entity)
            # if not str(node[2]).startswith('http://dbpedia.org/resource/'):
            #     print(node)
        else:
            assert parts[1] in ['beginIndex', 'label', 'endIndex', 'referenceContext', 'type']

    num_in = 0
    num_out = 0

    sorted_key = sorted(sent_char_index2tmp_entity.keys(), key=lambda x: (x[0], x[1], x[2]))
    doc_name2instance = dict()
    for (tmp_sent_index, char_start, char_end) in sorted_key:
        sentence = sentence_index2sentence[tmp_sent_index]
        if str(tmp_sent_index) not in doc_name2instance:
            doc_name2instance[str(tmp_sent_index)] = {
                'sentence': sentence,
                'entities': {
                    'starts': [],
                    'ends': [],
                    'entity_mentions': [],
                    'entity_names': [],
                }
            }
        tmp_entity = sent_char_index2tmp_entity[(tmp_sent_index, char_start, char_end)]
        processed_tmp_entity = tmp_entity.replace(' ', '_')
        if processed_tmp_entity in tmp_entity2entity:
            num_in += 1
            entity = tmp_entity2entity[processed_tmp_entity]
            # assert (char_end - char_start) == len(tmp_str)
            mention = sentence[char_start: char_end]
            doc_name2instance[str(tmp_sent_index)]['entities']['starts'].append(char_start)
            doc_name2instance[str(tmp_sent_index)]['entities']['ends'].append(char_end)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_mentions'].append(mention)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_names'].append(entity)

        else:
            num_out += 1
    print(f'num_in_kb: {num_in}; num_out_kb: {num_out}; len(tmp_entity2entity): {len(tmp_entity2entity)}')
    return doc_name2instance


def load_ttl_n3(
    file='/nfs/yding4/EL_project/dataset/n3-collection/Reuters-128.ttl',
):
    def process_sen_char(s):
        assert s.count('/') == 5
        first_parts = s.split('/')
        assert '#char=' in first_parts[-1]
        second_parts = first_parts[-1].split('#char=')
        assert len(second_parts) == 2
        assert ',' in second_parts[1]
        sentence_index = int(second_parts[0])
        third_parts = second_parts[1].split(',')
        assert len(third_parts) == 2
        char_start, char_end = int(third_parts[0]), int(third_parts[1])
        return char_start, char_end, sentence_index

    # file = '/nfs/yding4/EL_project/dataset/oke-challenge-2016/evaluation-data/task1/evaluation-dataset-task1.ttl'
    g = rdflib.Graph()
    g.parse(file, format='ttl')

    module_list = [
        'label', 'anchorOf', 'beginIndex', 'isString', 'sameAs', 'endIndex', 'taIdentRef', 'referenceContext', 'type', 'taSource', 'hasContext', 'sourceUrl'
    ]

    # 1. isString: extracts sentence (identified by the sentence number)
    # 2. taIdentRef: extracts mentions and labelled temporary annotations
    # 3. sameAs: bring temporary annotations to dataset-base if database has corresponding entities
    sentence_index2sentence = dict()
    sent_char_index2tmp_entity = dict()
    tmp_entity2entity = dict()
    num_in = 0
    num_out = 0

    for node_index, node in enumerate(g):
        # print(node)

        parts = node[1].split('#')
        assert len(parts) == 2
        if parts[1] not in module_list:
            print(str(parts[1]))
        assert parts[1] in module_list

        if parts[1] == 'anchorOf':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            # tmp_str = str(node[2]).rstrip()
            tmp_str = str(node[2])
            if (char_end - char_start) != len(tmp_str):
                # only one data error: 'Basel, Switzerland'
                tmp_str = tmp_str.split(',')[0]
            assert (char_end - char_start) == len(tmp_str)

        elif parts[1] == 'taIdentRef':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert '/' in str(node[2])
            tmp_entity = str(node[2]).split('/')[-1]
            assert (sentence_index, char_start, char_end) not in sent_char_index2tmp_entity
            if 'notInWiki' in str(node[2]):
                num_out += 1
            else:
                num_in += 1
                assert str(node[2].startswith('http://dbpedia.org/resource/'))

                sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] = urllib.parse.unquote(tmp_entity)

        elif parts[1] == 'isString':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert sentence_index not in sentence_index2sentence
            sentence_index2sentence[sentence_index] = str(node[2])

        elif parts[1] == 'sameAs':
            assert str(node[0]).count('sentence-') == 1
            mention = str(node[0]).split('sentence-')[1]

            entity = str(node[2]).split('/')[-1]
            if mention in tmp_entity2entity:
                assert entity == tmp_entity2entity[mention]
            tmp_entity2entity[mention] = entity
        else:
            if parts[1] == 'label':
                # 'label' is not useful
                tmp_split_str = str(node[0]).split('sentence-')[1]
                tmp_str = str(node[2])
                assert tmp_split_str == tmp_str.replace(' ', '_')

    sorted_key = sorted(sent_char_index2tmp_entity.keys(), key=lambda x: (x[0], x[1], x[2]))
    doc_name2instance = dict()
    for (tmp_sent_index, char_start, char_end) in sorted_key:
        sentence = sentence_index2sentence[tmp_sent_index]
        if str(tmp_sent_index) not in doc_name2instance:
            doc_name2instance[str(tmp_sent_index)] = {
                'sentence': sentence,
                'entities': {
                    'starts': [],
                    'ends': [],
                    'entity_mentions': [],
                    'entity_names': [],
                }
            }
        tmp_entity = sent_char_index2tmp_entity[(tmp_sent_index, char_start, char_end)]
        processed_tmp_entity = tmp_entity.replace(' ', '_')

        # assert (char_end - char_start) == len(tmp_str)
        mention = sentence[char_start: char_end]
        doc_name2instance[str(tmp_sent_index)]['entities']['starts'].append(char_start)
        doc_name2instance[str(tmp_sent_index)]['entities']['ends'].append(char_end)
        doc_name2instance[str(tmp_sent_index)]['entities']['entity_mentions'].append(mention)
        doc_name2instance[str(tmp_sent_index)]['entities']['entity_names'].append(processed_tmp_entity)

    # print(json.dumps(doc_name2instance, indent=4))
    print(f'num_in_kb: {num_in}; num_out_kb: {num_out};')

    return doc_name2instance


def gen_anno_from_xml(
        prefix='/nfs/yding4/In_Context_EL/data',
        dataset='ace2004',
        allow_mention_shift=False,
        allow_mention_without_entity=False,
        allow_repeat_annotation=False,
        has_prob=False,    # **YD-CL** whether ED probability exist in the xml
):

    """
    this function reads a standard xml EL annotation with its documents
    {dataset}:
    |
    |--RawText:
    |      |
    |      |---{doc_name} (with the txts)
    |
    |--{dataset}.xml (annotation of all the {doc_name})
    ATTENTION:
        a. the '&' is replaced as '&amp;' in both "txt" and "annotation" reading. '&' is not allowed in '.xml' file
        b. in the '.xml' annotation, 'doc_name' has no ' ', it has '_' instead. In mention and entity annotation, it has
        no '_' but has ' ' instead.
    :param prefix: the absolute path before the dataset directory.
    :param dataset: name of a dataset. It also forms the name of '.xml'.
    :param allow_mention_shift: allow mismatch between "txt[offset: offset + length]" and "{annotated mention}".
    If the flag is set to True: it will uses the length of "{annotated mention}" as actual length. Search the mention
    from "offset" - 10 to "offset + 100" to find this mention.
    If the flag is set to False: it will raise ERROR if a mismatch is found.
    :param allow_mention_without_entity: allow empty entity annotation, either '' or 'NIL', called "NER annotation".
    If the flag is set to True: the empty annotation will preserve, the entity will change to ''.
    If the flag is set to False: raise ERROR if an empty entity is found.
    :param allow_repeat_annotation: allow repeated annotation.
    If the flag is set to True: repeated annotation will not be considered as outputs
    If the flag is set to False: raise ERROR if a repeat annotation is found.
    :param has_prob
    If the flag is set to True: load ED probability
    :return:
    doc_name2txt, doc_name2anno:
    doc_name2txt: a dictionary of string. Each doc_name corresponds to a documentation of a dataset.
    doc_name2anno: a dictionary of list. Each doc_name corresponds to a documentation of a dataset.
    each element(ele) in the list is a dictionary formed with four elements:
    ele = {
            'start': offset,    # starting position of the mention in the doc_name txt.
            'end': offset + length, # endding position of the mention in the doc_name txt.
            'mention_txt': cur_mention, # annotated mention.
            'entity_txt': cur_ent_title, # annotated entity. '' or 'NIL' represents empty entity annotation (NER).
        }
    """

    raw_text_prefix = os.path.join(prefix, dataset + '/' + 'RawText')
    xml_file = os.path.join(prefix, dataset + '/' + dataset + '.xml')
    doc_names = os.listdir(raw_text_prefix)

    # collect documentation for each doc_name
    doc_name2txt = dict()

    for doc_name in doc_names:
        txt_path = os.path.join(raw_text_prefix, doc_name)
        txt = ''
        with open(txt_path, 'r') as reader:
            for line in reader:
                txt += line
        doc_name2txt[doc_name] = txt.replace('&amp;', '&')

    # collect mention/entity annotation from xml
    doc_name2anno = defaultdict(list)
    # nested named entity recognition problem in silver + gold
    reader = open(xml_file, 'r')

    doc_str_start = 'document docName=\"'
    doc_str_end = '\">'

    line = reader.readline()
    num_el_anno = 0
    num_ner_anno = 0
    num_shift_anno = 0
    num_change_length = 0
    cur_doc_name = ''

    while line:
        if doc_str_start in line:
            start = line.find(doc_str_start)
            end = line.find(doc_str_end)
            cur_doc_name = line[start + len(doc_str_start): end]
            cur_doc_name = cur_doc_name.replace('&amp;', '&').replace(' ', '_')
            assert cur_doc_name in doc_name2txt

            # **YD** preserve empty annotation for a doc_name
            doc_name2anno[cur_doc_name]

        else:
            if '<annotation>' in line:
                line = reader.readline()

                # **YD** bug here because mention may contain line changing symbols, aka. annotated mention cross more
                # than one line.
                # assert '<mention>' in line and '</mention>' in line

                assert '<mention>' in line
                new_line = line
                while '</mention>' not in new_line:
                    new_line = reader.readline()
                    line += new_line

                m_start = line.find('<mention>') + len('<mention>')
                m_end = line.find('</mention>')

                cur_mention = line[m_start: m_end]
                cur_mention = cur_mention.replace('&amp;', '&').replace('_', ' ')

                line = reader.readline()
                # assert '<wikiName>' in line and '</wikiName>' in line
                e_start = line.find('<wikiName>') + len('<wikiName>')
                e_end = line.find('</wikiName>')
                cur_ent_title = '' if '<wikiName/>' in line else line[e_start: e_end]
                cur_ent_title = cur_ent_title.replace('&amp;', '&').replace('_', ' ')

                line = reader.readline()
                assert '<offset>' in line and '</offset>' in line
                off_start = line.find('<offset>') + len('<offset>')
                off_end = line.find('</offset>')
                offset = int(line[off_start: off_end])

                line = reader.readline()
                assert '<length>' in line and '</length>' in line
                len_start = line.find('<length>') + len('<length>')
                len_end = line.find('</length>')
                length_record = int(line[len_start: len_end])
                length = len(cur_mention)

                if length != length_record:
                    print('mention', cur_mention, 'offset', offset, 'length', length, 'length_record', length_record)
                    num_change_length += 1

                if has_prob:    # **YD-CL** whether ED probability exist in the xml.
                    line = reader.readline()
                    assert '<prob>' in line and '</prob>' in line
                    prob_start = line.find('<prob>') + len('<prob>')
                    prob_end = line.find('</prob>')
                    prob = float(line[prob_start: prob_end])

                line = reader.readline()
                if '<entity/>' in line:
                    line = reader.readline()

                if not has_prob:    # **YD-CL** allow 'prob' in the xml, but not loading it
                    assert '</annotation>' in line

                # if cur_ent_title != 'NIL' and cur_ent_title != '':
                assert cur_doc_name != ''
                ele = {
                        'start': offset,
                        'end': offset + length,
                        'mention_txt': cur_mention,
                        'entity_txt': cur_ent_title,
                    }
                if has_prob:    # **YD-CL** whether ED probability exist in the xml.
                    ele['prob'] = prob

                doc_txt = doc_name2txt[cur_doc_name]
                pos_mention = doc_txt[offset: offset + length]

                if allow_mention_shift:
                    if pos_mention != ele['mention_txt']:
                        num_shift_anno += 1
                        offset = max(0, offset - 10)
                        while pos_mention != cur_mention:
                            offset = offset + 1
                            pos_mention = doc_txt[offset: offset + length]
                            if offset > ele['start'] + 100:
                                print(
                                    'pos_mention',
                                    doc_txt[anno['offset']: anno['offset'] + length],
                                    anno['mention_txt'],
                                )
                                raise ValueError('huge error!')
                        ele['start'] = offset
                        ele['end'] = offset + length
                else:
                    if pos_mention != ele['mention_txt']:
                        print('pos_mention', pos_mention)
                        print("ele['mention_txt']", ele['mention_txt'])
                    assert pos_mention == ele['mention_txt'], 'Unmatched mention between annotation mention ' \
                                                              'and annotation position'

                # allow_mention_without_entity
                if ele['entity_txt'] == '' or ele['entity_txt'] == 'NIL':
                    ele['entity_txt'] = ''

                # consider repeat annotations in wikipedia
                if ele not in doc_name2anno[cur_doc_name]:
                    if ele['entity_txt'] != '':
                        doc_name2anno[cur_doc_name].append(ele)
                        num_el_anno += 1
                    else:
                        num_ner_anno += 1
                        if allow_mention_without_entity:
                            doc_name2anno[cur_doc_name].append(ele)
                else:
                    if not allow_repeat_annotation:
                        raise ValueError('find repeated annotation: ' + str(ele))

        line = reader.readline()

    print(
        'num_ner_anno', num_ner_anno,
        'num_el_anno', num_el_anno,
        'num_shift_anno', num_shift_anno,
        'num_change_length', num_change_length
    )

    # **YD** post-processing: sort the annotation by start and end.
    for doc_name in doc_name2anno:
        tmp_anno = doc_name2anno[doc_name]
        tmp_anno = sorted(tmp_anno, key=lambda x: (x['start'], x['end']))
        doc_name2anno[doc_name] = tmp_anno

    # transform original format into the same format.
    doc_name2instance = dict()
    for doc_name in doc_name2anno:
        assert doc_name in doc_name2txt
        txt = doc_name2txt[doc_name]
        anno = doc_name2anno[doc_name]
        starts = []
        ends = []
        entity_mentions = []
        entity_names = []

        for entity_instance in anno:
            start = entity_instance['start']
            end = entity_instance['end']
            entity_mention = entity_instance['mention_txt']
            entity_name = entity_instance['entity_txt']
            starts.append(start)
            ends.append(end)
            entity_mentions.append(entity_mention)
            entity_names.append(entity_name)
        
        if len(starts) == 0:
            continue
        doc_name2instance[doc_name] = {
            'sentence': txt,
            'entities': {
                'starts': starts,
                'ends': ends,
                'entity_mentions':entity_mentions ,
                'entity_names': entity_names,
            }
        }

    return doc_name2instance

    '''
    doc_name2instance[str(tmp_sent_index)] = {
                'sentence': sentence,
                'entities': {
                    'starts': [],
                    'ends': [],
                    'entity_mentions': [],
                    'entity_names': [],
                }
            }
    '''


def dataset_loader(file, key='', mode='tsv'): 
    '''
    file: input dataset file
    key: only used for aida, to consider train/valid/test split
    mode: options to consider different types of input file
    # mode to be expanded to multiple ED datasets
    '''
    if mode == 'tsv':
        doc_name2instance = load_tsv(file, key=key)
    elif mode == 'oke_2015':
        doc_name2instance = load_ttl_oke_2015(file)
    elif mode == 'oke_2016':
        doc_name2instance = load_ttl_oke_2016(file)
    elif mode == 'n3':
        doc_name2instance = load_ttl_n3(file)
    elif mode == 'unseen_mentions':
        doc_name2instance = load_unseen_mentions(file)
    elif mode == 'xml':
        parent_dir = os.path.dirname(os.path.dirname(file))
        dataset = os.path.basename(file).split('.')[0]
        doc_name2instance = gen_anno_from_xml(prefix=parent_dir, dataset=dataset)
    else:
        raise ValueError('unknown mode!')
    return doc_name2instance


if __name__ == '__main__':
    # load_tsv()
    # load_ttl_oke_2015()
    # load_ttl_oke_2016()
    # load_ttl_n3('/nfs/yding4/EL_project/dataset/n3-collection/Reuters-128.ttl')
    # load_ttl_n3('/nfs/yding4/EL_project/dataset/n3-collection/RSS-500.ttl')

    # file = '/nfs/yding4/e2e_EL_evaluate/data/wned/xml/ori_xml2revise_xml/ace2004/ace2004.xml'
    # doc_name2instance = dataset_loader(file, mode='xml')
    doc_name2instance = load_unseen_mentions()