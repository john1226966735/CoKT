"""
 This is a question-level variant of DKT created by us, it corresponding to CoKT-Embed in our paper.
 Author: Zhou Jianpeng
"""

from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
DEVICE = None


class DKT_QuesEmb(nn.Module, ABC):
    def __init__(self, args, data):
        super(DKT_QuesEmb, self).__init__()
        global DEVICE
        DEVICE = torch.device(args.device)

        self.input_module = InputModule(data['num_ques'], args.emb_dim)
        self.ks_module = KnowledgeStateModule(args.rnn_mode, self.input_module.input_dim, args.hidden_dim,
                                              args.rnn_num_layer)
        self.predict_module = PredictModule(args.hidden_dim, args.emb_dim, args.exercise_dim)

    def forward(self, seq_lens, pad_curr, pad_answer, pad_next):
        # get interact/question embedding
        interact_emb, next_ques_emb = self.input_module(pad_curr, pad_answer, pad_next)
        # update knowledge state
        ks_emb = self.ks_module(interact_emb)
        # predict
        pad_predict = self.predict_module(ks_emb, next_ques_emb)
        pack_predict = pack_padded_sequence(pad_predict, seq_lens, enforce_sorted=True)
        return pack_predict


class InputModule(nn.Module, ABC):
    def __init__(self, num_ques, ques_emb_dim):
        super(InputModule, self).__init__()
        self.input_dim = 2 * ques_emb_dim
        # for question embedding
        self.ques_emb_layer = nn.Embedding(num_ques, ques_emb_dim)

        # for fusing question embedding and correctness
        self.transform_matrix = torch.zeros(2, self.input_dim, device=DEVICE)
        self.transform_matrix[0][ques_emb_dim:] = 1.0
        self.transform_matrix[1][:ques_emb_dim] = 1.0

    def forward(self, pad_curr, pad_answer, pad_next):  # [seq, bs]
        # get embedding of question
        curr_ques_emb = self.ques_emb_layer(pad_curr)  # get current questions' embeddings, [seq, bs, dim]
        next_ques_emb = self.ques_emb_layer(pad_next)  # get predict questions' embeddings, [seq, bs, dim]

        # concatenate zero vector in front of or behind curr_ques_emb according to correctness
        answer_emb = F.embedding(pad_answer, self.transform_matrix)
        interact_emb = torch.cat((curr_ques_emb, curr_ques_emb), -1) * answer_emb
        return interact_emb, next_ques_emb


class KnowledgeStateModule(nn.Module, ABC):
    def __init__(self, rnn_mode, input_dim, hidden_dim, num_layer):
        super(KnowledgeStateModule, self).__init__()
        assert rnn_mode in ['lstm', 'rnn', 'gru']
        if rnn_mode == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layer, batch_first=False)
        elif rnn_mode == 'rnn':
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layer, batch_first=False)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layer, batch_first=False)

    def forward(self, pad_interact_emb):
        pad_ks_emb, _ = self.rnn(pad_interact_emb)
        return pad_ks_emb


class PredictModule(nn.Module, ABC):
    def __init__(self, ks_dim, question_dim, exercise_dim):
        super(PredictModule, self).__init__()
        self.h2y = nn.Linear(ks_dim + question_dim, exercise_dim)
        self.y2o = nn.Linear(exercise_dim, 1)

    def forward(self, ks_emb, question_emb):
        y = F.relu(self.h2y(torch.cat((ks_emb, question_emb), -1)))
        prediction = torch.sigmoid(self.y2o(y)).squeeze(-1)
        return prediction
