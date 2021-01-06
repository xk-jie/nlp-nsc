import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer

from configs.config import Config

config = Config.yaml_config()

class NscDataset(Dataset):
    def __init__(self, df, mode='train'):
        super(NscDataset, self).__init__()
        # datasource, dataframe
        self.df = df
        self.mode = mode
        self.sentences = self.df['sentence'].values
        self.ate_labels = self.df['ate_label'].values

        self.bert_tokenizer = BertTokenizer.from_pretrained(config['bert_name'])
        self.max_seq_len = config['max_seq_len']

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]

        encode_dict = self.bert_tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            truncation='longest_first',
            max_length=self.max_seq_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encode_dict['input_ids'].flatten()
        attention_mask = encode_dict['attention_mask'].flatten()
        token_type_ids = encode_dict['token_type_ids'].flatten()

        assert len(input_ids) == self.max_seq_len, \
            f'Error with input length {len(input_ids)} vs {self.max_seq_len}'
        assert len(attention_mask) == self.max_seq_len, \
            f'Error with input length {len(attention_mask)} va {self.max_seq_len}'
        assert len(token_type_ids) == self.max_seq_len, \
            f'Error with input length {len(token_type_ids)} vs {self.max_seq_len}'

        ret = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

        if self.mode != 'test':
            ate_label = self.ate_labels[item]
            ret['ate_label'] = torch.tensor(ate_label, dtype=torch.long)

        return ret

def dataloader_generator(df, mode='train'):
    shuffle = False
    if mode == 'train':
        shuffle = True

    dataset = NscDataset(df, mode)

    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config['batch_size']
    )
