import pandas as pd

from configs.config import Config

config = Config.yaml_config()

def read_file(filename):
    ate_label_map = {word: i for i, word in enumerate(['O', 'B-ASP', 'I-ASP', '[CLS]', '[SEP]'], 1)}

    data, sentence, ate_label = [], [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.split()
            if len(cols) > 0 and len(cols) != 3:
                print(f'warning. detected error line {line} in file.')
                continue

            if cols:
                sentence.append(cols[0])

                temp_key = cols[1]
                if temp_key not in ate_label_map.keys():
                    temp_key = list(ate_label_map.keys())[0]
                ate_label.append(ate_label_map[temp_key])
            else:
                if len(sentence) > 0:
                    # add [CLS],[SEP] to ate_label
                    ate_label.insert(0, ate_label_map['[CLS]'])
                    ate_label.append(ate_label_map['[SEP]'])
                    # fill with padding
                    max_seq_len = config['max_seq_len']
                    if len(ate_label) < max_seq_len:
                        ate_label = ate_label + [0] * (max_seq_len - len(ate_label))
                    elif len(ate_label) > max_seq_len:
                        ate_label = ate_label[:max_seq_len]

                    data.append((sentence, ate_label))
                sentence = []
                ate_label = []

    df = pd.DataFrame(data=data, columns=['sentence', 'ate_label'])

    return df

def load_data():
    df_train = read_file('corpus/data/twitter/twitter.atepc.train.dat')
    df_test = read_file('corpus/data/twitter/twitter.atepc.test.dat')

    if 'corpus_size' in config.keys():
        corpus_size = config['corpus_size']
        return df_train[:corpus_size], df_test[:corpus_size]

    return df_train, df_test
