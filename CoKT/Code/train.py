import torch
import torch.nn as nn
from Code.CoKT import CoKT
from Code.DKT_QE import DKT_QuesEmb
from Code.utils import Logger
from sklearn import metrics
import numpy as np


def train(loader, args):
    logger = Logger(args)

    device = torch.device(args.device)
    assert args.model_type in ['DKT_QE', 'CoKT']
    if args.model_type == 'DKT_QE':
        model = DKT_QuesEmb(args, loader).to(device)
    else:
        model = CoKT(args, loader).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    criterion = nn.BCELoss(reduction='mean')

    for epoch in range(1, args.max_epoch + 1):
        logger.epoch_increase()
        for i, (seq_lens, pad_data, pad_answer, pad_index, pack_label) in enumerate(loader['train']):
            pack_predict = model(seq_lens, pad_data, pad_answer, pad_index)
            loss = criterion(pack_predict.data, pack_label.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics_dict = evaluate(model, loader['train'])
        test_metrics_dict = evaluate(model, loader['test'])

        logger.one_epoch(epoch, train_metrics_dict, test_metrics_dict, model)

        if logger.is_stop():
            break
        # end of epoch
    logger.one_run(args)
    # end of run


def evaluate(model, data):
    model.eval()
    true_list, pred_list = [], []
    for seq_lens, pad_data, pad_answer, pad_index, pack_label in data:
        pack_pred = model(seq_lens, pad_data, pad_answer, pad_index)

        y_true = pack_label.data.cpu().contiguous().view(-1).detach()
        y_pred = pack_pred.data.cpu().contiguous().view(-1).detach()

        true_list.append(y_true)
        pred_list.append(y_pred)
    auc = metrics.roc_auc_score(np.concatenate(true_list, 0), np.concatenate(pred_list, 0))
    model.train()
    return {'auc': auc}
