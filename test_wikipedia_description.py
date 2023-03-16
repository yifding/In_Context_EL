from REL.wikipedia import Wikipedia
from REL.wikipedia_yago_freq import WikipediaYagoFreq

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


# 1. load dataset
input_file = '/nfs/yding4/EL_project/dataset/KORE50/AIDA.tsv'
doc_name2instance = load_tsv(input_file)

base_url ="/nfs/yding4/REL/data/"
wiki_version="wiki_2014"
wikipedia = Wikipedia(base_url, wiki_version)
wikipedia_yago_freq = WikipediaYagoFreq(base_url, wiki_version, wikipedia)
wikipedia_yago_freq.extract_entity_description()

for doc_name, instance in doc_name2instance.items():
    entities = instance['entities']
    for entity_name in entities['entity_names']:
        entity_id = wikipedia.ent_wiki_id_from_name(entity_name)
        print(entity_name, entity_id)
        if entity_id in wikipedia_yago_freq.entity_id2description:
            entity_description = wikipedia_yago_freq.entity_id2description[entity_id]
            print(f"entity_description: {entity_description[:200]}")