#!/bin/bash

export PATH=/nfs/yding4/conda_envs/pair_query/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/pair_query/lib:$LD_LIBRARY_PATH

CODE=/nfs/yding4/In_Context_EL/in_context_el/in_context_ed/evaluation.py
INPUT_FILE="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/multi_choice_prompt/oke_2016.json"
DATASET="oke_2016"
OUTPUT_DIR="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/evaluation"
BASE_URL="/nfs/yding4/REL/data/"
WIKI_VERSION="wiki_2014"
NUM_ENTITY_CANDIDATES=10
NUM_ENTITY_DESCRIPTION_CHARACTERS=150

python3 ${CODE} \
    --input_file ${INPUT_FILE}    \
    --dataset ${DATASET}    \
    --output_dir ${OUTPUT_DIR}  \
    --base_url ${BASE_URL}  \
    --wiki_version ${WIKI_VERSION}  

