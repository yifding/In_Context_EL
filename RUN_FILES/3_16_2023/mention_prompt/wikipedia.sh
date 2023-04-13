#!/bin/bash

export PATH=/nfs/yding4/conda_envs/pair_query/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/pair_query/lib:$LD_LIBRARY_PATH

CODE=/nfs/yding4/In_Context_EL/in_context_el/in_context_ed/collect_prompt.py
MODE="xml"
INPUT_FILE="/nfs/yding4/e2e_EL_evaluate/data/wned/xml/ori_xml2revise_xml/wikipedia/wikipedia.xml"
OUTPUT_FILE="wikipedia.json"
OUTPUT_DIR="/nfs/yding4/In_Context_EL/RUN_FILES/3_16_2023/mention_prompt"

python3 ${CODE} \
    --input_file ${INPUT_FILE}    \
    --output_dir ${OUTPUT_DIR}   \
    --output_file ${OUTPUT_FILE}   \
    --mode  ${MODE}

