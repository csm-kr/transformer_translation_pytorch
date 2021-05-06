import os
import re
import pandas as pd
from pathlib import Path
import pickle
import torch

from torchtext.legacy import data as ttd
from torchtext.legacy.data import Example, Dataset


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


if __name__ == '__main__':
    train_data, valid_data = load_dataset(mode='train')
    train_iter, valid_iter = make_iter(10, 'train', train_data=train_data, valid_data=valid_data)
    print(train_iter)