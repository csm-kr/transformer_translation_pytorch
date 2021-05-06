from functools import reduce

from torch.nn import ModuleList
import torch.nn.functional as F
import copy
import math
import torch
import numpy as np
"""
ModuleList는 목록에 하위 모듈을 보관하것
이때 모듈들은 파이썬 리스트들 처럼 인덱스를 사용할 수 있다.
"""
def clones(module, N):
  return ModuleList([copy.deepcopy(module) for i in range(N)])

"""
디코더에서 어텐션 스코어 매트릭스에서
이후의 값들에 대해 -∞으로 마스킹 처리해주기 위한 함수
(1, size, size)의 마스크를 리턴한다.
"""
def subsequent_mask(size):
  "Mask out subsequent positions."
  attn_shape = (1, size, size)
  subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
  return torch.from_numpy(subsequent_mask) == 0

def log(t, eps=1e-9):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

def temperature_sampling(logits, temperature):
  if temperature is None or temperature == 0.0:
    return torch.argmax(logits)
  probs = F.softmax(logits / temperature)
  pred_ids = probs.cpu().multinomial(probs.size()[1], replacement=False)
  return pred_ids


import os
import re
import json
import pickle
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

from torchtext.legacy import data as ttd
from torchtext.legacy.data import Example, Dataset


def load_dataset(mode):
    """
    Load train, valid and test dataset as a pandas DataFrame
    Args:
        mode: (string) configuration mode used to which dataset to load
    Returns:
        (DataFrame) train, valid, test dataset converted to pandas DataFrame
    """
    print(f'Loading AI Hub Kor-Eng translation dataset and converting it to pandas DataFrame . . .')

    data_dir = Path().cwd() / 'data'

    if mode == 'train':
        train_file = os.path.join(data_dir, 'train.csv')
        train_data = pd.read_csv(train_file, encoding='utf-8')

        valid_file = os.path.join(data_dir, 'valid.csv')
        valid_data = pd.read_csv(valid_file, encoding='utf-8')

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(valid_data)}')

        return train_data, valid_data

    else:
        test_file = os.path.join(data_dir, 'test.csv')
        test_data = pd.read_csv(test_file, encoding='utf-8')

        print(f'Number of testing examples: {len(test_data)}')

        return test_data


def clean_text(text):
    """
    remove special characters from the input sentence to normalize it
    Args:
        text: (string) text string which may contain special character
    Returns:
        normalized sentence
    """
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`…》]', '', text)
    return text


def convert_to_dataset(data, kor, eng):
    """
    Pre-process input DataFrame and convert pandas DataFrame to torchtext Dataset.
    Args:
        data: (DataFrame) pandas DataFrame to be converted into torchtext Dataset
        kor: torchtext Field containing Korean sentence
        eng: torchtext Field containing English sentence
    Returns:
        (Dataset) torchtext Dataset containing 'kor' and 'eng' Fields
    """
    # drop missing values not containing str value from DataFrame
    missing_rows = [idx for idx, row in data.iterrows() if type(row.korean) != str or type(row.english) != str]
    data = data.drop(missing_rows)

    # convert each row of DataFrame to torchtext 'Example' containing 'kor' and 'eng' Fields
    list_of_examples = [Example.fromlist(row.apply(lambda x: clean_text(x)).tolist(),
                                         fields=[('kor', kor), ('eng', eng)]) for _, row in data.iterrows()]

    """
    tokenize 한 결과들이 들어있는 list
    list_of_examples[0].eng
    ['i', 'do', "n't", 'think', 'we', 'love', 'each', 'other', 'like', 'before']
    list_of_examples[0].kor
    ['우리', '가', '예전처럼', '좋아하', '고', '있지', '않다고', '생각', '해']
    
    list_of_examples[1].kor
    ['경제', '가', '내년에', '회복될', '거라고', '해요']
    list_of_examples[2].kor
    ['좋습니다', '그럼', '근처', '일식당으로', '제가', '예약하겠습니다']
    """
    # construct torchtext 'Dataset' using torchtext 'Example' list

    dataset = Dataset(examples=list_of_examples, fields=[('kor', kor), ('eng', eng)])
    return dataset


def make_iter(batch_size, mode, train_data=None, valid_data=None, test_data=None):
    """
    Convert pandas DataFrame to torchtext Dataset and make iterator which will be used to train and test the model
    Args:
        batch_size: (integer) batch size used to make iterators
        mode: (string) configuration mode used to which iterator to make
        train_data: (DataFrame) pandas DataFrame used to build train iterator
        valid_data: (DataFrame) pandas DataFrame used to build validation iterator
        test_data: (DataFrame) pandas DataFrame used to build test iterator
    Returns:
        (BucketIterator) train, valid, test iterator
    """
    # load text and label field made by build_pickles.py
    file_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(file_kor)

    file_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(file_eng)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # convert pandas DataFrame to torchtext dataset
    if mode == 'train':
        train_data = convert_to_dataset(train_data, kor, eng)
        valid_data = convert_to_dataset(valid_data, kor, eng)

        # make iterator using train and validation dataset
        print(f'Make Iterators for training . . .')
        train_iter, valid_iter = ttd.BucketIterator.splits(
            (train_data, valid_data),
            # the BucketIterator needs to be told what function it should use to group the data.
            # In our case, we sort dataset using text of example
            sort_key=lambda sent: len(sent.kor),
            # all of the tensors will be sorted by their length by below option
            sort_within_batch=True,
            batch_size=batch_size,
            device=device)

        return train_iter, valid_iter

    else:
        test_data = convert_to_dataset(test_data, kor, eng)

        # defines dummy list will be passed to the BucketIterator
        dummy = list()

        # make iterator using test dataset
        print(f'Make Iterators for testing . . .')
        test_iter, _ = ttd.BucketIterator.splits(
            (test_data, dummy),
            sort_key=lambda sent: len(sent.kor),
            sort_within_batch=True,
            batch_size=batch_size,
            device=device)

        return test_iter


def epoch_time(start_time, end_time):
    """
    Calculate the time spent to train one epoch
    Args:
        start_time: (float) training start time
        end_time: (float) training end time
    Returns:
        (int) elapsed_mins and elapsed_sec spent for one epoch
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def display_attention(candidate, translation, attention):
    """
    displays the model's attention over the source sentence for each target token generated.
    Args:
        candidate: (list) tokenized source tokens
        translation: (list) predicted target translation tokens
        attention: a tensor containing attentions scores
    Returns:
    """
    # attention = [target length, source length]

    attention = attention.cpu().detach().numpy()
    # attention = [target length, source length]

    font_location = 'pickles/NanumSquareR.ttf'
    fontprop = fm.FontProperties(fname=font_location)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.matshow(attention, cmap='bone')
    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + [t.lower() for t in candidate], rotation=45, fontproperties=fontprop)
    ax.set_yticklabels([''] + translation, fontproperties=fontprop)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


class Params:
    """
    Class that loads hyperparameters from a json file
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)
        self.load_vocab()

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def load_vocab(self):
        # load kor and eng vocabs to add vocab size configuration
        pickle_kor = open('pickles/kor.pickle', 'rb')
        kor = pickle.load(pickle_kor)

        pickle_eng = open('pickles/eng.pickle', 'rb')
        eng = pickle.load(pickle_eng)

        # add device information to the the params
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # add <sos> and <eos> tokens' indices used to predict the target sentence
        params = {'input_dim': len(kor.vocab), 'output_dim': len(eng.vocab),
                  'sos_idx': eng.vocab.stoi['<sos>'], 'eos_idx': eng.vocab.stoi['<eos>'],
                  'pad_idx': eng.vocab.stoi['<pad>'], 'device': device}

        self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


if __name__ == '__main__':
    param = Params('config/params.json')