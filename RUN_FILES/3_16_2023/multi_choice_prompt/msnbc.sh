#!/bin/bash

export PATH=/nfs/yding4/conda_envs/pair_query/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/pair_query/lib:$LD_LIBRARY_PATH

CODE=/nfs/yding4/In_Context_EL/in_context_el/in_context_ed/collect_multi_choice_prompt.py
INPUT_FILE="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/mention_prompt/msnbc.json"
OUTPUT_FILE="msnbc.json"
OUTPUT_DIR="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/multi_choice_prompt"
BASE_URL="/nfs/yding4/REL/data/"
WIKI_VERSION="wiki_2014"
NUM_ENTITY_CANDIDATES=10
NUM_ENTITY_DESCRIPTION_CHARACTERS=150

python3 ${CODE} \
    --input_file ${INPUT_FILE}    \
    --output_dir ${OUTPUT_DIR}  \
    --output_file ${OUTPUT_FILE}    \
    --base_url ${BASE_URL}  \
    --wiki_version ${WIKI_VERSION}  \
    --num_entity_candidates ${NUM_ENTITY_CANDIDATES}    \
    --num_entity_description_characters ${NUM_ENTITY_DESCRIPTION_CHARACTERS}    

