import numpy as np
import pandas as pd

from results.submit import predict_to_file
from models.metrics import(
    map,
    mrr,
    search_mcc
)
from utils.helper import id_to_label
from configs.config import Config

config = Config.yaml_config()

def search_metrics(labels, predictions):
    acc = (labels == predictions).sum() / len(labels)
    map_score = map(labels, predictions)
    mrr_score = mrr(labels, predictions)
    mcc_score = search_mcc(labels, predictions)

    print(f'Accuracy is {acc}, MAP is {map_score}, MRR is {mrr_score}, MCC is {mcc_score}')

def do_result(df, preds):
    data = []
    for index, row in df.iterrows():
        predictions = preds[index]
        label = row['ate_label']
        sentence = row['sentence']

        for i, word in enumerate(sentence):
            try:
                lab = id_to_label(label[i + 1])
                prediction = id_to_label(predictions[i + 1])

                data.append((word, lab, prediction))
            except:
                pass

    df2 = pd.DataFrame(data=data, columns=config['submit_csv_columns'])

    predict_to_file(df2, mode='csv')
