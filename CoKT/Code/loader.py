import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence

import os
import numpy as np

DEVICE = None


def load_data(args):
    filePath_dict, dataList_dict, dataset_dict, dataLoader_dict = dict(), dict(), dict(), dict()
    shuffle = {'train': True, 'test': False}
    global DEVICE
    DEVICE = torch.device(args.device)
    for train_or_test in ['train', 'test']:
        filePath_dict[train_or_test] = os.path.join(args.data_path, args.dataset, train_or_test + '_ques.txt')
        dataList_dict[train_or_test] = file_to_list(filePath_dict[train_or_test], args.min_seq_len, args.max_seq_len)
        dataset_dict[train_or_test] = KTDataset(dataList_dict[train_or_test][0], dataList_dict[train_or_test][1], dataList_dict[train_or_test][2])
        dataLoader_dict[train_or_test] = DataLoader(dataset_dict[train_or_test], batch_size=args.batch_size, collate_fn=collate_fn, shuffle=shuffle[train_or_test])

    # Specific for CoKT: Load pre-trained question embeddings
    # hyper-params for node2vec
    walk_length = '80'
    num_walks = '10'
    window_size = '5'
    epoch = '20'
    p = '1.00'
    q = '1.00'
    pre_emb_list = []
    for g in args.used_graphs:
        param_list = [args.dataset, 'kg_'+g, str(args.emb_dim), str(walk_length), num_walks, window_size, epoch, p, q]
        pre_emb_path = '%s/%s.embQ.npy' % (args.pre_emb_dir, '_'.join(param_list))
        pre_emb_list.append(torch.from_numpy(np.load(pre_emb_path)).to(DEVICE))
    dataLoader_dict['pre_emb_list'] = pre_emb_list
    dataLoader_dict['num_ques'] = pre_emb_list[0].size(0)

    print('load data done!')
    return dataLoader_dict


def file_to_list(filename, min_seq_len, max_seq_len):
    def split_func(_seq_len):
        _split_list = []
        while _seq_len > 0:
            if _seq_len >= max_seq_len:
                _split_list.append(max_seq_len)
                _seq_len -= max_seq_len
            elif _seq_len >= min_seq_len:
                _split_list.append(_seq_len)
                _seq_len -= _seq_len
            else:
                _seq_len -= min_seq_len
        return len(_split_list), _split_list

    seq_lens, ques_ids, answers = [], [], []
    k_split = -1
    with open(filename) as file:
        lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if i % 3 == 0:
            seq_len = int(line)
            if seq_len < min_seq_len:
                i += 3
                continue
            else:
                k_split, split_list = split_func(seq_len)
                seq_lens += split_list
        else:
            line = line.split(',')
            array = [int(e) for e in line]
            if i % 3 == 1:
                for j in range(k_split):
                    ques_ids.append(array[max_seq_len * j: max_seq_len * (j + 1)])
            else:
                for j in range(k_split):
                    answers.append(array[max_seq_len * j: max_seq_len * (j + 1)])
        i += 1
    # for integrity, check the lengths
    assert len(seq_lens) == len(ques_ids) == len(answers)
    return seq_lens, ques_ids, answers


class KTDataset(Dataset):
    def __init__(self, seq_lens, ques_ids, answers):
        self.seq_lens = seq_lens
        self.ques_ids = ques_ids
        self.answers = answers

    def __len__(self):
        return len(self.seq_lens)

    def __getitem__(self, item):
        seq_len = self.seq_lens[item]
        ques_id = self.ques_ids[item]
        answer = self.answers[item]

        sample_len = torch.tensor([seq_len - 1], dtype=torch.long)
        sample_exercise = torch.tensor(ques_id[:-1], dtype=torch.long)
        sample_answer = torch.tensor(answer[:-1], dtype=torch.long)
        sample_next_exercise = torch.tensor(ques_id[1:], dtype=torch.long)
        sample_next_answer = torch.tensor(answer[1:], dtype=torch.float)
        return sample_len, sample_exercise, sample_answer, sample_next_exercise, sample_next_answer


def collate_fn(batch):
    # Sort the batch in the descending order
    batch = sorted(batch, key=lambda x: x[0], reverse=True)

    seq_lens = torch.cat([x[0] for x in batch])
    exercises = pad_sequence([x[1] for x in batch], batch_first=False)
    answers = pad_sequence([x[2] for x in batch], batch_first=False)
    next_exercises = pad_sequence([x[3] for x in batch], batch_first=False)
    next_answers = pack_sequence([x[4] for x in batch], enforce_sorted=True)
    return seq_lens.to(DEVICE), exercises.to(DEVICE), answers.to(DEVICE), next_exercises.to(DEVICE), next_answers.to(
        DEVICE)
