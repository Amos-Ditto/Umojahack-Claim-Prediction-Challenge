import tensorflow as tf
import time
import os
import re
import pandas as pd
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab


path = os.getcwd()
Root_dir = os.path.abspath(os.path.join(path, os.pardir))

path_train = os.path.join(Root_dir, 'data/train.csv')

df = pd.read_csv(path_train)

df = df.drop(columns=['policy_number', 'loss_date', 'claim_id', 'claim_number'])
x_df = df[['product','agent','class_of_business','risk_type','client_type','renewal_frequency',
    'primary_cause','secondary_cause','branch'
]]
x_train = x_df.to_numpy()

x_data = tf.data.Dataset.from_tensor_slices(x_train)


bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ['[PAD]', '[UNK]']

bert_vocab_args = dict(
    vocab_size = 8000,
    reserved_tokens=reserved_tokens,
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={},
)


text_vocab = bert_vocab.bert_vocab_from_dataset(
    x_data.batch(64).prefetch(2),
    **bert_vocab_args
)

def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)


write_vocab_file('vocabulary.txt', text_vocab)