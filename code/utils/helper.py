import random
import datetime
import regex as re
import numpy as np
import pandas as pd

import torch

from configs.config import Config

config = Config.yaml_config()

def set_seed_cudnn(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def show_dataframe(df):
    # show all dataframe data
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)

def format_time(elapsed):
    # takes a time in seconds
    # round to the nearest second
    elapsed_round = int(round(elapsed))
    # format hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_round))

def clean_str(string):
    """
    Tokenization/stringing cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.stringip().lower()

def id_to_label(*args):
    label_list = config['ate_label_list']
    map = {i: label for i, label in enumerate(label_list, 1)}
    reverse_map = dict([value, key] for key, value in map.items())

    if len(args) == 1:
        label = map.get(args[0], 'O')

        return label

    if len(args) == 2:
        label1, label2 = [], []
        for i, id in enumerate(args[0]):
            # 去掉[CLS]的值，因为在计算交叉熵时去除了第一个值
            if id == reverse_map.get('[CLS]', -1):
                continue
            label1.append(map.get(id, 'O'))
            label2.append(map.get((args[1])[i], 'O'))

        return label1, label2

    return None
