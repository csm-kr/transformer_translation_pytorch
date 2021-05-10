import torch
from torch.utils.data import DataLoader, Dataset
import csv
import os
from utils import clean_text, subsequent_mask
from tqdm import tqdm
from transformers import BertTokenizer
from torch.autograd import Variable


class KorEngDataset(Dataset):
    def __init__(self,
                 root='../data',
                 split='train',
                 tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-cased'),
                 max_length=64):

        super(KorEngDataset, self).__init__()

        assert split in ['train', 'test', 'valid']
        self.file_name = os.path.join(root, split) + '.csv'
        self.docs = []

        pad_token_idx = tokenizer.pad_token_id

        # data 준비 and clean up
        with open(self.file_name, 'r', encoding='UTF-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            print(csv_reader)

            self.lines = []
            for i, line in enumerate(csv_reader):
                if i == 0:
                    continue
                kor_text = clean_text(line[0])
                eng_text = clean_text(line[1])
                self.lines.append([kor_text, eng_text])

        # data encode 하여서
        for line in tqdm(self.lines):
            input = tokenizer.encode(line[0], max_length=max_length, truncation=True)
            rest = max_length - len(input)
            input = torch.tensor(input + [pad_token_idx] * rest)

            target = tokenizer.encode(line[1], max_length=max_length, truncation=True)
            rest = max_length - len(target)
            target = torch.tensor(target + [pad_token_idx] * rest)

            doc = {
                'input_str': tokenizer.convert_ids_to_tokens(input),
                'input': input,  # input
                'input_mask': (input != pad_token_idx).unsqueeze(-2),       # input_mask
                'target_str': tokenizer.convert_ids_to_tokens(target),
                'target': target,  # target,
                'target_mask': self.make_std_mask(target, pad_token_idx),   # target_mask
                'token_num': (target[..., 1:] != pad_token_idx).data.sum()  # token_num - pad 아닌 token 의 갯수
            }
            self.docs.append(doc)

    @staticmethod
    def make_std_mask(tgt, pad_token_idx):
        target_mask = (tgt != pad_token_idx).unsqueeze(-2)
        # make look-ahead mask - 하나씩 늘려가면서
        target_mask = target_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(target_mask.data))
        return target_mask.squeeze()

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        item = self.docs[idx]
        return item

    def collate_fn(self, batch):

        input_str = list()
        input = list()
        input_mask = list()
        target_str = list()
        target = list()
        target_mask = list()
        token_num = list()

        for b in batch:
            input_str.append(b['input_str'])
            input.append(b['input'])
            input_mask.append(b['input_mask'])
            target_str.append(b['target_str'])
            target.append(b['target'])
            target_mask.append(b['target_mask'])
            token_num.append(b['token_num'])

        return {
            "input_str": input_str,                                                               # input_str
            "input": torch.stack(input).contiguous(),                                             # input
            "input_mask": torch.stack(input_mask).contiguous(),                                   # input_mask
            "target_str": target_str,                                                             # target_str
            "target": torch.stack(target).contiguous(),                                           # target,
            "target_mask": torch.stack(target_mask).contiguous(),                                 # target_mask
            "token_num": torch.stack(token_num).contiguous(),                                     # token_num
        }


if __name__ == '__main__':
    train_set = KorEngDataset(split='valid')
    train_loader = DataLoader(dataset=train_set,
                              batch_size=32,
                              shuffle=True,
                              collate_fn=train_set.collate_fn)

    for i, data in enumerate(train_loader):

        input = data['input']                         # [B, 64]
        target = data['target']                       # [B, 64]
        input_mask = data['input_mask']               # [B, 1, 64]
        target_mask = data['target_mask']             # [B, 64, 64] ??

        print(input.size())
        print(target.size())
        print(input_mask.size())
        print(target_mask.size())

