import os
import json
from in_context_el.dataset_reader import dataset_loader


def generate_standard_EL():
    datasets = ['KORE50', 'msnbc', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500', 'aida_test', 'derczynski']
    dataset_files = [
        '/nfs/yding4/EL_project/dataset/KORE50/AIDA.tsv',
        '/nfs/yding4/e2e_EL_evaluate/data/wned/xml/ori_xml2revise_xml/msnbc/msnbc.xml',
        '/nfs/yding4/EL_project/dataset/oke-challenge/evaluation-data/task1/evaluation-dataset-task1.ttl',
        '/nfs/yding4/EL_project/dataset/oke-challenge-2016/evaluation-data/task1/evaluation-dataset-task1.ttl',
        '/nfs/yding4/EL_project/dataset/n3-collection/Reuters-128.ttl',
        '/nfs/yding4/EL_project/dataset/n3-collection/RSS-500.ttl',
        '/nfs/yding4/e2e_EL_evaluate/data/aida/xml/xml_from_end2end_neural_el/aida_testb/aida_testb.xml',
        '/nfs/yding4/EL_project/dataset/derczynski/ipm_nel_corpus/correct_ipm_nel.conll',
    ]

    dataset_modes = [
        'tsv',
        'xml',
        'oke_2015',
        'oke_2016',
        'n3',
        'n3',
        'xml',
        'derczynski',
    ]

    output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/datasets'
    os.makedirs(output_dir, exist_ok=True)

    assert len(datasets) == len(dataset_files) == len(dataset_modes)

    for dataset, dataset_file, dataset_mode in zip(
        datasets, dataset_files, dataset_modes):
        output_file = os.path.join(output_dir, dataset + '.json')
        print(f'dataset: {dataset}; dataset_file: {dataset_file}; dataset_mode: {dataset_mode}')
        doc_name2instance = dataset_loader(dataset_file, mode=dataset_mode)
        with open(output_file, 'w') as writer:
            json.dump(doc_name2instance, writer, indent=4)


def generate_standard_ED():
    datasets = ['aida_test', 'msnbc', 'aquaint', 'ace2004', 'clueweb', 'wikipedia']
    dataset_files = [
        '/nfs/yding4/e2e_EL_evaluate/data/aida/xml/xml_from_end2end_neural_el/aida_testb/aida_testb.xml',
        '/nfs/yding4/e2e_EL_evaluate/data/wned/xml/ori_xml2revise_xml/msnbc/msnbc.xml',
        '/nfs/yding4/e2e_EL_evaluate/data/wned/xml/ori_xml2revise_xml/msnbc/aquaint.xml',
        '/nfs/yding4/e2e_EL_evaluate/data/wned/xml/ori_xml2revise_xml/msnbc/ace2004.xml',
        '/nfs/yding4/e2e_EL_evaluate/data/wned/xml/ori_xml2revise_xml/msnbc/clueweb.xml',
        '/nfs/yding4/e2e_EL_evaluate/data/wned/xml/ori_xml2revise_xml/msnbc/wikipedia.xml',
    ]

    dataset_modes = [
        'xml',
        'xml',
        'xml',
        'xml',
        'xml',
        'xml',
    ]

    output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/ED_standard_datasets'
    os.makedirs(output_dir, exist_ok=True)

    assert len(datasets) == len(dataset_files) == len(dataset_modes)

    for dataset, dataset_file, dataset_mode in zip(
        datasets, dataset_files, dataset_modes):
        output_file = os.path.join(output_dir, dataset + '.json')
        print(f'dataset: {dataset}; dataset_file: {dataset_file}; dataset_mode: {dataset_mode}')
        doc_name2instance = dataset_loader(dataset_file, mode=dataset_mode)
        with open(output_file, 'w') as writer:
            json.dump(doc_name2instance, writer, indent=4)



def generate_additional_ED():
    datasets = ['KORE50', 'oke_2015', 'oke_2016', 'Reuters-128', 'RSS-500']
    dataset_files = [
        '/nfs/yding4/EL_project/dataset/KORE50/AIDA.tsv',
        '/nfs/yding4/EL_project/dataset/oke-challenge/evaluation-data/task1/evaluation-dataset-task1.ttl',
        '/nfs/yding4/EL_project/dataset/oke-challenge-2016/evaluation-data/task1/evaluation-dataset-task1.ttl',
        '/nfs/yding4/EL_project/dataset/n3-collection/Reuters-128.ttl',
        '/nfs/yding4/EL_project/dataset/n3-collection/RSS-500.ttl',
    ]

    dataset_modes = [
        'tsv',
        'oke_2015',
        'oke_2016',
        'n3',
        'n3',
    ]

    output_dir = '/nfs/yding4/In_Context_EL/RUN_FILES/11_14_2024/ED_addtional_datasets'
    os.makedirs(output_dir, exist_ok=True)

    assert len(datasets) == len(dataset_files) == len(dataset_modes)

    for dataset, dataset_file, dataset_mode in zip(
        datasets, dataset_files, dataset_modes):
        output_file = os.path.join(output_dir, dataset + '.json')
        print(f'dataset: {dataset}; dataset_file: {dataset_file}; dataset_mode: {dataset_mode}')
        doc_name2instance = dataset_loader(dataset_file, mode=dataset_mode)
        with open(output_file, 'w') as writer:
            json.dump(doc_name2instance, writer, indent=4)


if __name__ == '__main__':
    # generate_standard_ED()
    generate_additional_ED()