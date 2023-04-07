#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu # specify the queue
#$-l gpu_card=4
#$-N end2end_neural_EL_server

export PATH=/home/yding4/anaconda3/envs/TF1.4/bin:$PATH
export LD_LIBRARY_PATH=/home/yding4/anaconda3/envs/TF1.4/lib:$LD_LIBRARY_PATH

DIR=/nfs/yding4/e2e_EL_evaluate/baseline/end2end_neural_el/code

cd ${DIR}

# el script
# CUDA_VISIBLE_DEVICES="" python -m gerbil.server --training_name=base_att_global --experiment_name=paper_models \
#     --persons_coreference_merge=True --all_spans_training=True --entity_extension=extension_entities


CUDA_VISIBLE_DEVICES="" python -m gerbil.server --training_name=base_att_global --experiment_name=paper_models \
    --persons_coreference_merge=True --ed_mode --entity_extension=extension_entities