import os
import json
from tqdm import tqdm
import openai
openai.api_key = "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"

import random
from REL.wikipedia import Wikipedia
from REL.wikipedia_yago_freq import WikipediaYagoFreq
from REL.mention_detection import MentionDetection

# base_url ="/nfs/yding4/REL/data/"
# wiki_version="wiki_2014"
# mention_detection = MentionDetection(base_url, wiki_version)

# wikipedia = Wikipedia(base_url, wiki_version)
# wikipedia_yago_freq = WikipediaYagoFreq(base_url, wiki_version, wikipedia)
# wikipedia_yago_freq.extract_entity_description()

# mention = 'Klum'
# mention = 'Brad'
# processed_mention = mention_detection.preprocess_mention(mention)
# candidate_entities = mention_detection.get_candidates(processed_mention)
# print(candidate_entities)

# input_file = '/nfs/yding4/pair_query/in_context/RUN_FILES/init_prompt/KORE50.json'
# with open(input_file) as reader:
#     doc_name2instance = json.load(reader)

for doc_name, instance in tqdm(doc_name2instance.items()):
    entities = instance['entities']
    entity_mentions = entities['entity_mentions']
    entity_names = entities['entity_names']
    prompt_results = entities['prompt_result']
    multi_selection_prompts = []
    multi_selection_prompt_results = []
    ground_truth_entity_index = []
    for (
        entity_mention, 
        entity_name, 
        prompt_result,
    ) in zip(
        entity_mentions, 
        entity_names, 
        prompt_results,
    ):
        processed_mention = mention_detection.preprocess_mention(entity_mention)
        candidate_entities = mention_detection.get_candidates(processed_mention)
        processed_candidate_entities = [wikipedia.preprocess_ent_name(candidate_entity[0]) for candidate_entity in candidate_entities[:10]]
        processed_entity_name = wikipedia.preprocess_ent_name(entity_name)
        gt_index = -1
        if processed_entity_name in processed_candidate_entities:
            gt_index = processed_candidate_entities.index(processed_entity_name)
        else:
            len_candidate_entities = len(processed_candidate_entities)
            gt_index = random.randint(0, len_candidate_entities)
            processed_candidate_entities = processed_candidate_entities[:gt_index] + [processed_entity_name] + processed_candidate_entities[gt_index:]
        ground_truth_entity_index.append(gt_index)
        multi_selection_prompt = ''
        for index, processed_candidate_entity in enumerate(processed_candidate_entities):
            id = wikipedia.ent_wiki_id_from_name(processed_candidate_entity)
            if id in wikipedia_yago_freq.entity_id2description:
                description = wikipedia_yago_freq.entity_id2description[id][:150]
            else:
                description = processed_candidate_entity
        
            multi_selection_prompt += f'({index + 1}). ' + description + '\n'
        
        multi_selection_prompt = prompt_result + '\n\n' + f'Which of the following entity best describe {entity_mention} ?' + '\n\n' + multi_selection_prompt
        openai_output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": multi_selection_prompt},
            ]
        )
        complete_output = openai_output["choices"][0]["message"]['content']
        multi_selection_prompts.append(multi_selection_prompt)
        multi_selection_prompt_results.append(complete_output)
        # break   # for fast test

    doc_name2instance[doc_name]['multi_selection_prompts'] = multi_selection_prompts
    doc_name2instance[doc_name]['multi_selection_prompt_results'] = multi_selection_prompt_results
    doc_name2instance[doc_name]['ground_truth_entity_index'] = ground_truth_entity_index
    # break


output_file = '/nfs/yding4/pair_query/in_context/RUN_FILES/init_prompt/multi_choice_KORE50.json'
with open(output_file, 'w') as writer:
    json.dump(doc_name2instance, writer, indent=4)