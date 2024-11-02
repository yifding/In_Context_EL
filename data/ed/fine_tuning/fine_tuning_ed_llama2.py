import os
import replicate
from in_context_el.openai_key import REPLICATE_API_KEY
os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_KEY

training = replicate.trainings.create(
    version="meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00",
    input={
    # set your inputs here
    'train_data': 'https://replicate.delivery/pbxt/KBtoctgd3fLrIDrtPuR4WZazk7yxc1TfxWTjVVXzyzVEp4Jl/aida_train_gpt.jsonl',
    'validation_data': 'https://replicate.delivery/pbxt/KBttFdTpvw916kLt5Hk8YXTZjoB7P2VIGTgNJf5sMT6qdyvC/aida_testa_gpt.jsonl',
    'train_batch_size': 8,
    'num_train_epochs': 1,
    },
    destination="yifding/aida_discriminative"
)

print(training)