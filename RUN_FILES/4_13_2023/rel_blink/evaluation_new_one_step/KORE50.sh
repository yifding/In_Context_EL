#!/bin/bash

export PATH=/nfs/yding4/conda_envs/pair_query/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/pair_query/lib:$LD_LIBRARY_PATH

CODE=/nfs/yding4/In_Context_EL/in_context_el/in_context_ed/evaluation_raw.py
INPUT_FILE="/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/new_one_step_prompt/KORE50.json"
OUTPUT_DIR="/nfs/yding4/In_Context_EL/RUN_FILES/4_13_2023/rel_blink/evaluation_new_one_step"
DATASET="KORE50"

python3 ${CODE} \
    --input_file ${INPUT_FILE}    \
    --output_dir ${OUTPUT_DIR}   \
    --dataset ${DATASET}