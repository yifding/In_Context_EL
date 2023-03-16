


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


if __name__ == '__main__':
    load_tsv()
    load_ttl_oke_2015()
    load_ttl_oke_2016()
    load_ttl_n3('/nfs/yding4/EL_project/dataset/n3-collection/Reuters-128.ttl')
    load_ttl_n3('/nfs/yding4/EL_project/dataset/n3-collection/RSS-500.ttl')