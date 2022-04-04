import matplotlib.pyplot as plt
import os
import time
import torch


class Logger:
    def __init__(self, config):
        self.n_epoch = 0
        self.start_timestamp = time.time()
        self.train_auc_list = []
        self.test_auc_list = []
        self.best_metric_dict = {'auc': 0.0, 'acc': 0.0}
        self.best_epoch = -1
        self.save_dir = config.save_dir
        self.log_file = config.log_file
        self.result_file = config.result_file
        self.patience = config.patience
        self.duration = ''
        self.end_time = ''
        self.log = ''

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        self.pic_dir = self.save_dir + '/' + "pic"
        if not os.path.isdir(self.pic_dir):
            os.mkdir(self.pic_dir)
        self.log_writer = open(os.path.join(self.save_dir, self.log_file), 'a', encoding='utf-8')
        self.result_writer = open(os.path.join(self.save_dir, self.result_file), 'a', encoding='utf-8')
        self.param_path = '../param/params_%s_%d_%d_%d.pkl' % (config.dataset, config.emb_dim, config.hidden_dim, config.exercise_dim)

    def one_epoch(self, epoch, train_metric_dict, test_metric_dict, model):
        self.log = 'epoch=%d,train=%s,test=%s\n' % (epoch, train_metric_dict, test_metric_dict)
        self._logWriter()

        if self.best_metric_dict['auc'] < test_metric_dict['auc']:
            self.best_metric_dict = test_metric_dict.copy()
            self.best_epoch = epoch
            # torch.save(model.state_dict(), self.param_path)

        self._aucAppend(train_metric_dict['auc'], test_metric_dict['auc'])

    def one_run(self, args):
        self._getTime()
        # self._draw()  # for saving disk space, invalid it as default

        result_dict = {'t': self.end_time, 'duration': self.duration, 'n_epoch': self.n_epoch, 'best_epoch': self.best_epoch}
        result_dict.update(self.best_metric_dict)
        result_dict.update(vars(args))

        self._resultWriter("%s\n" % str(result_dict))

    def _aucAppend(self, train_auc, test_auc):
        self.train_auc_list.append(train_auc)
        self.test_auc_list.append(test_auc)

    def is_stop(self):
        if len(self.test_auc_list) < self.patience:
            return False
        array = self.test_auc_list[-self.patience - 1:]
        if max(array[1:]) >= array[0]:
            return False
        else:
            return True

    def _logWriter(self):
        print(self.log.rstrip('\n'))
        # self.log_writer.write(log)  # for saving disk memory, invalid this code as default

    def _resultWriter(self, result):
        print(result.rstrip('\n'))
        self.result_writer.write(result)
        # self.log_writer.write(result)

    def _draw(self):
        plt.figure()
        plt.plot(range(len(self.train_auc_list)), self.train_auc_list, label='train_auc', marker='o')
        plt.plot(range(len(self.test_auc_list)), self.test_auc_list, label='test_auc', marker='s')
        plt.title('%s' % self.end_time)
        plt.xlabel('epoch')
        plt.ylabel('auc')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.pic_dir, '%s.png' % self.end_time))
        plt.close()

    def _getTime(self):
        self.end_time = '%s' % time.strftime("%Y-%m-%d&%H-%M-%S", time.localtime(time.time()))
        s = time.time() - self.start_timestamp
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        self.duration = "%d:%d:%d" % (h, m, s)

    def epoch_increase(self):
        self.n_epoch += 1
