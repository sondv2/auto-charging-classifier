import os
import numpy as np
import pandas as pd


def save_to_txt(data, file_path):
    with open(file_path, 'w') as output:
        output.writelines('\n'.join(data))


def load_txt(file_path):
    with open(file_path, 'r') as input:
        return input.readlines()


def save_to_csv(data, file_path, **kwargs):
    data.to_csv(file_path, index=False, **kwargs)


def load_csv(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)


def load_json(file_path, **kwargs):
    return pd.read_json(file_path, **kwargs)


def readlines(file_path):
    with open(file_path, 'r', encoding='utf8') as finp:
        for line in finp:
            yield line


def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def get_file_name(file_path):
    return file_path.split('/')[-1].split('\\')[-1].split('.')[0]

