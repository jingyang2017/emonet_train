import logging
import os
import time
from typing import List
import torch
from utils.evaluation import evaluate_flip
from utils.utils_logging import AverageMeter
from utils.metrics import CCC, PCC, RMSE, SAGR, ACC


class CallBackEvaluation(object):
    def __init__(self,test_dataloader_no_flip,test_dataloader_flip,subset='valid'):
        self.metrics_valence_arousal = {'CCC': CCC, 'PCC': PCC, 'RMSE': RMSE, 'SAGR': SAGR}
        self.metrics_expression = {'ACC': ACC}
        self.test_dataloader_no_flip = test_dataloader_no_flip
        self.test_dataloader_flip = test_dataloader_flip
        self.highest_acc: float = 0.0
        self.best_epoch: int = 0
        self.subset = subset


    def __call__(self, epoch,model):
        results={}
        valence_results, arousal_results, acc_expressions = \
            evaluate_flip(model, self.test_dataloader_no_flip, self.test_dataloader_flip,
                      metrics_valence_arousal=self.metrics_valence_arousal, metrics_expression=self.metrics_expression)
        results['ACC'] = acc_expressions['ACC']
        if acc_expressions['ACC'] > self.highest_acc:
            self.highest_acc = acc_expressions['ACC']
            self.best_epoch = epoch
        logging.info('%s [%d]Accuracy-Highest: %.5f [%d] Accuracy-Current:%.3f' % (self.subset, self.best_epoch, self.highest_acc, epoch, acc_expressions['ACC']))
        logging.info('%s current valence'%(self.subset))
        for key, value in valence_results.items():
            results['valence_'+key]=value
            logging.info('%s/%.3f'%(key,value))
        # logging.info('\n')
        logging.info('%s current arousal'%(self.subset))
        for key,value in arousal_results.items():
            results['arousal_'+key]=value
            logging.info('%s/%.3f'%(key,value))
        # logging.info('\n')
        return results


class CallBackVerification(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank is 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank is 0 and num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, writer=None):
        self.frequent: int = frequent
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.writer = writer
        self.init = False
        self.tic = 0

    def __call__(self, global_step, loss,loss_pcc,acc, epoch, opt):
        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed
                except ZeroDivisionError:
                    speed = float('inf')
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)

                msg = "Speed %.2f samples/sec Loss %.4f AU %.4f top1 %.4f Epoch: %d Global Step: %d LR1: %.5f Time: %1.f hours" % (
                    speed_total, loss.avg, loss_pcc.avg, acc.avg, epoch, global_step, opt.param_groups[0]['lr'], time_for_end)
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


class CallBackModelCheckpoint(object):
    def __init__(self,output="./"):
        self.output: str = output

    def __call__(self, epoch, backbone: torch.nn.Module):
        torch.save(backbone.module.state_dict(), os.path.join(self.output, "backbone_%d.pth"%epoch))

