from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from Code.DKT_QE import DKT_QuesEmb


class CoKT(DKT_QuesEmb, ABC):
    def __init__(self, args, data):
        super(CoKT, self).__init__(args, data)
        self.input_module.ques_emb_layer = LoadFusePretrainEmb(args, data)


class LoadFusePretrainEmb(nn.Module, ABC):
    def __init__(self, args, data):
        super(LoadFusePretrainEmb, self).__init__()
        self.num_graph = len(args.used_graphs)
        self.pre_emb_list = data['pre_emb_list']

        if self.num_graph > 1:
            concat_dim = sum([emb.size(-1) for emb in self.pre_emb_list])
            self.fuse_emb_layer = nn.Linear(concat_dim, args.emb_dim)

    def forward(self, pad_ques):
        # get embedding of present question
        batch_emb_list = []
        for emb_mat in self.pre_emb_list:
            batch_emb_list.append(F.embedding(pad_ques, emb_mat))
        if self.num_graph > 1:
            fused_ques_emb = F.relu(self.fuse_emb_layer(torch.cat(batch_emb_list, dim=-1)))
        else:
            fused_ques_emb = batch_emb_list[0]
        return fused_ques_emb

