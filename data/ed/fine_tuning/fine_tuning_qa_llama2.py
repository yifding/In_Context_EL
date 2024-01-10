import os
import replicate

os.environ['REPLICATE_API_TOKEN'] = 'r8_EzgWgZmHLxJ7JP0NI2V5HnyOyQkHydW1lkFPP'

training = replicate.trainings.create(
    version="meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00",
    input={
    # set your inputs here
    'train_data': 'https://replicate.delivery/pbxt/KCIPSJKGF2qARP9YM54Kmv5VoOBQiT73LkzeJhpfb3X83kal/finetune_data_final.jsonl',
    'train_batch_size': 8,
    'num_train_epochs': 1,
    },
    destination="yifding/aida_qa_llama2"
)

print(training)