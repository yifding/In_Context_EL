#!/bin/bash

export PATH=/nfs/yding4/conda_envs/pair_query/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/pair_query/lib:$LD_LIBRARY_PATH

CODE=/nfs/yding4/In_Context_EL/in_context_el/in_context_ed/collect_prompt.py
MODE="oke_2015"
INPUT_FILE="/nfs/yding4/EL_project/dataset/oke-challenge/evaluation-data/task1/evaluation-dataset-task1.ttl"
OUTPUT_FILE="oke_2015.json"
OUTPUT_DIR="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/mention_prompt"

python3 ${CODE} \
    --input_file ${INPUT_FILE}    \
    --output_dir ${OUTPUT_DIR}   \
    --output_file ${OUTPUT_FILE}   \
    --mode  ${MODE}

