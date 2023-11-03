import os
import json
import random
import jsonlines
from torch.utils.data import Dataset


class TypingDataset(Dataset):
    def __init__(
            self,
            mode='train',
            input_data_dir='/afs/crc.nd.edu/user/y/yding4/ET_project/In_Context_EL/RUN_FILES/11_2_2023/nli_second_step',
    ):
        self.data = []

        output_fine2ultrafine = os.path.join(input_data_dir, 'fine2ultrafine.json')
        output_general2fine = os.path.join(input_data_dir, 'general2fine.json')
        output_train_file = os.path.join(input_data_dir, 'train_file.json')
        output_dev_file = os.path.join(input_data_dir, 'dev_file.json')
        output_test_file = os.path.join(input_data_dir, 'test_file.json')

        with open(output_fine2ultrafine) as reader:
            fine2ultrafine = json.load(reader)
        with open(output_general2fine) as reader:
            general2fine = json.load(reader)

        self.general_label_list = sorted(general2fine.keys())
        self.fine_label_list = sorted(fine2ultrafine.keys())
        ultrafine_label_list = []
        for tmp_value in fine2ultrafine.values():
            ultrafine_label_list.extend(tmp_value)
        self.ultrafine_label_list = sorted(ultrafine_label_list)
        self.label_lst = self.general_label_list + self.fine_label_list + self.ultrafine_label_list
        if mode == 'train':
            data_file = output_train_file
        elif mode == 'dev':
            data_file = output_dev_file
        elif mode == 'test':
            data_file = output_test_file
        else:
            raise ValueError('Unknown split, please choose from train, dev, test')
        with jsonlines.open(data_file) as reader:
            for line in reader:
                premise = line['text'][:250]
                annotation = [line['l1'], line['l2'], line['l3']]
                annotation_general = [line['l1']]
                annotation_fine = [line['l2']]
                annotation_ultrafine = [line['l3']]
                self.data.append([premise, annotation, annotation_general, annotation_fine, annotation_ultrafine])

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)