"""
1. 基础库
2. 神经网络框架库
3. 神经网络应用库
4. 本地库
"""
import time
import numpy as np
from sklearn.model_selection import KFold

from utils.helper import (
    set_seed,
    format_time
)
from corpus import corpus
from configs.config import Config
from datasets.dataset import dataloader_generator
from results.result import do_result
from models.nsc import NscModel

# global config
config = Config.yaml_config()
# global corpus data
df_train, df_test = corpus.load_data()
print('df_train.shape:', df_train.shape)
print('df_test.shape:', df_test.shape)

def kf_data():
    # The same group will not appear in two different folds
    # (the number of distinct groups has to be at least equal to the number of folds).
    # split: Generate indices to split data into training and test set
    group_kfold = KFold(n_splits=config['k'], shuffle=True)
    gfk = group_kfold.split(X=df_train)

    return gfk

def kfold_train():
    # calculate the train time
    t0 = time.time()

    eval_preds, test_preds = np.zeros((len(df_train), 1)), []
    # GroupKFold train
    for k, (train_idx, eval_idx) in enumerate(kf_data()):
        print(f'fold train, {k} in {config["k"]}')
        train_dataloader = dataloader_generator(df_train.iloc[train_idx])
        eval_dataloader = dataloader_generator(df_train.iloc[eval_idx], mode='eval')

        # init model
        model = NscModel()
        # train, evaluate
        model.fit(train_dataloader, eval_dataloader)

        # eval predict
        ## eval_pred = model.predict(eval_dataloader)
        ## eval_preds[eval_idx] = eval_pred

        # test predict
        test_dataloader = dataloader_generator(df_test, mode='test')
        test_pred = model.predict(test_dataloader)
        test_preds.append(test_pred)

    total_time = format_time(time.time() - t0)
    print(f'train finished, took: {total_time}')

    return np.array(eval_preds), np.array(test_preds)

def main():
    # Set the seed value all over the place to make this reproducible.
    set_seed(config['seed'])
    # get preds by train
    _, test_preds = kfold_train()

    avg_test_preds = np.average(test_preds, axis=0)

    test_preds = np.argmax(avg_test_preds, 2)
    test_preds = test_preds.reshape(-1, config['max_seq_len'])

    # generate result
    do_result(df_test, test_preds)

if __name__ == '__main__':
    main()
