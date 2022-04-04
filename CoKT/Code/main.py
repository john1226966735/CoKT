import argparse
from Code.train import train
from Code.loader import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    # tunable hyper-parameters: emb_dim, hidden_dim, l2_weight, lr, fine_tune_emb
    # for loading data
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--data_path", type=str, default="../Data")
    parser.add_argument("--dataset", type=str, default='ASSIST09')
    parser.add_argument("--pre_emb_dir", type=str, default='./node2vec/emb')
    parser.add_argument("--used_graphs", type=list, default=['cw', 'pk'])

    # for embedding and input layer
    parser.add_argument("--emb_dim", type=int, default=128)

    # for knowledge state layer
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--rnn_mode", type=str, default='lstm')
    parser.add_argument("--rnn_num_layer", type=int, default=1)

    # for predict layer
    parser.add_argument("--exercise_dim", type=int, default=128)

    # for training
    parser.add_argument("--model_type", type=str, default='CoKT')  # CoKT, DKT_QE
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--l2_weight", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument('--device', type=str, default="cuda:1")

    parser.add_argument('--save_dir', type=str, default='../Result/ASSIST09',
                        help='the dir which save results')
    parser.add_argument('--log_file', type=str, default='logs.txt',
                        help='the name of logs file')
    parser.add_argument('--result_file', type=str, default='results.txt',
                        help='the name of results file')
    parser.add_argument('--remark', type=str, default='',
                        help='remark the experiment')
    return parser.parse_args()


# ---------- main ----------
# used_graphs can be set as ['cw', 'pk'], ['cw'] and ['pk'],
# respectively corresponding to CoKT, CoKT-NoSkill and CoKT-NoStu in our paper.
args = parse_args()
args.dataset = "ASSIST09"
args.save_dir = "../Result/" + args.dataset
data_loader = load_data(args)
train(data_loader, args)
