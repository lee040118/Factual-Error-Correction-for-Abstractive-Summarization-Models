import json
import os
import glob
import torch
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader, IterableDataset

def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data
#
class KoBARTSummaryDataset(Dataset):
    def __init__(self,filepath, tok, max_len, pad_index = 0, ignore_index=-100):
        super().__init__()
        self.tok = tok
        self.max_len = max_len
        # self.docs = pd.read_csv(filepath, sep='\n', header=None)
        # self.docs = pd.read_csv(filepath, sep='\t')
        self.docs = load_json(filepath)
        self.docs = pd.DataFrame(self.docs, columns=["text", "summary", "corrupt_sum", "id", "label", "augmentation_span"])
        self.len = len(self.docs)
        self.pad_index = pad_index
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        if type(instance['corrupt_sum']) == str:
            instance['corrupt_sum'] = instance['corrupt_sum'][:instance['augmentation_span'][0]] + self.tok.mask_token + instance['corrupt_sum'][instance['augmentation_span'][1]:]
            input_ids = [self.tok.bos_token]+  self.tok.tokenize(instance['corrupt_sum']) + [self.tok.eos_token]
        else:
            input_ids = [self.tok.bos_token]+ self.tok.tokenize(instance['summary']) + [self.tok.eos_token]

        input_ids += [self.tok.eos_token]+ self.tok.tokenize(instance['text'])+ [self.tok.eos_token]
        input_ids = self.add_padding_data(input_ids)
        input_ids = self.tok.convert_tokens_to_ids(input_ids)

        label_ids = self.tok.encode(instance['summary'])
        label_ids.append(self.tok.eos_token_id)

        dec_input_ids = [self.pad_index]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)

        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}

    def __len__(self):
        return self.len