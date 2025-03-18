import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from torch.distributions import Categorical
from TCN_Learner import TCN_Learner
from copy import deepcopy
import argparse

class HHI_Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        super(HHI_Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.new_lr = args.meta_lr * args.focus_lr
        self.net = TCN_Learner(config)

        if args.meta_optim == 'AdamW':
            self.meta_optim = optim.AdamW(self.net.parameters(), lr=self.meta_lr)
        elif args.meta_optim == 'Adam':
            self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.meta_optim, lr_lambda=lambda epoch: args.lr_decay ** int(epoch))
        self.test_acc = -1

    def clip_grad_by_norm_(self, grad, max_norm):
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)
        return total_norm / counter

    def focal_loss(self, predictions, targets, alpha, gamma):
        # Ensure predictions have the correct shape
        predictions = predictions.view(-1, predictions.size(-1))

        ce_loss = F.cross_entropy(predictions, targets.view(-1), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weights = (alpha * (1 - pt) ** gamma)
        focal_loss = torch.mean(focal_weights * ce_loss)

        return focal_loss

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        # 在這裡實現微調過程的程式碼
        # x_spt 是支援集，y_spt 是對應的標籤
        # x_qry 是查詢集，y_qry 是對應的標籤
        # 返回微調後的模型和相關結果
        task_num = x_spt.shape[0]
        pre_label = []
        true_label = []
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]
        for i in range(task_num):
            loss = 0
            for idx in range(x_spt[i].shape[0]):
                logits = self.net(x_spt[i][idx], vars=None, bn_training=True, dropout_training=True).to(torch.float)
                loss += self.focal_loss(logits, y_spt[i][idx], self.alpha, self.gamma)
            loss /= x_spt[i].shape[0]
            grad = torch.autograd.grad(loss, self.net.parameters(), allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            with torch.no_grad():
                logits_q = self.net(x_qry[i][0], self.net.parameters(), bn_training=True, dropout_training=False)
                loss_q = self.focal_loss(logits_q, y_qry[i][0], self.alpha, self.gamma)
                losses_q[0] += loss_q
                pred_q = logits_q.argmax(dim=0)
                correct = torch.eq(pred_q, y_qry[i][0]).sum().item()
                corrects[0] = corrects[0] + correct
            with torch.no_grad():
                logits_q = self.net(x_qry[i][0], fast_weights, bn_training=True, dropout_training=False)
                loss_q = self.focal_loss(logits_q, y_qry[i][0], self.alpha, self.gamma)
                losses_q[1] += loss_q
                pred_q = logits_q.argmax(dim=0)
                correct = torch.eq(pred_q, y_qry[i][0]).sum().item()
                corrects[1] = corrects[1] + correct
            for k in range(1, self.update_step):
                loss = 0
                for idx in range(x_spt[i].shape[0]):
                    logits = self.net(x_spt[i][idx], fast_weights, bn_training=True, dropout_training=True)
                    loss += self.focal_loss(logits, y_spt[i][idx], self.alpha, self.gamma)
                    
                loss /= x_spt[i].shape[0]
                grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = self.net(x_qry[i][0], fast_weights, bn_training=True, dropout_training=False)
                loss_q = self.focal_loss(logits_q, y_qry[i][0], self.alpha, self.gamma)
                losses_q[k + 1] += loss_q
                with torch.no_grad():
                    pred_q = logits_q.argmax(dim=0)
                    if k == self.update_step - 1:
                        pre_label.append(pred_q.item())
                        true_label.append(y_qry[i][0])
                    correct = torch.eq(pred_q, y_qry[i][0]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct
        loss_q = losses_q[-1] / task_num
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        return corrects, losses_q[-1], pre_label, true_label

    def forward(self, x_spt, y_spt, x_qry, y_qry, dropout_training=True):
        task_num = x_spt.shape[0]
        pre_label = []
        true_label = []
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]
        for i in range(task_num):
            loss = 0
            for idx in range(x_spt[i].shape[0]):
                logits = self.net(x_spt[i][idx], vars=None, bn_training=True, dropout_training=dropout_training).to(torch.float)
                loss += self.focal_loss(logits, y_spt[i][idx], self.alpha, self.gamma)
                
            loss /= x_spt[i].shape[0]
            grad = torch.autograd.grad(loss, self.net.parameters(), allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            with torch.no_grad():
                logits_q = self.net(x_qry[i][0], self.net.parameters(), bn_training=True, dropout_training=False)
                loss_q = self.focal_loss(logits_q, y_qry[i][0], self.alpha, self.gamma)
                losses_q[0] += loss_q
                pred_q = logits_q.argmax(dim=0)
                correct = torch.eq(pred_q, y_qry[i][0]).sum().item()
                corrects[0] = corrects[0] + correct
            with torch.no_grad():
                logits_q = self.net(x_qry[i][0], fast_weights, bn_training=True, dropout_training=False)
                loss_q = self.focal_loss(logits_q, y_qry[i][0], self.alpha, self.gamma)
                losses_q[1] += loss_q
                pred_q = logits_q.argmax(dim=0)
                correct = torch.eq(pred_q, y_qry[i][0]).sum().item()
                corrects[1] = corrects[1] + correct
            for k in range(1, self.update_step):
                loss = 0
                for idx in range(x_spt[i].shape[0]):
                    logits = self.net(x_spt[i][idx], fast_weights, bn_training=True, dropout_training=dropout_training)
                    loss += self.focal_loss(logits, y_spt[i][idx], self.alpha, self.gamma)
                loss /= x_spt[i].shape[0]
                grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                
                logits_q = self.net(x_qry[i][0], fast_weights, bn_training=True, dropout_training=False)
                loss_q = self.focal_loss(logits_q, y_qry[i][0], self.alpha, self.gamma)
                losses_q[k + 1] += loss_q
                with torch.no_grad():
                    pred_q = logits_q.argmax(dim=0)
                    if k == self.update_step - 1:
                        pre_label.append(pred_q.item())
                        true_label.append(y_qry[i][0])
                    correct = torch.eq(pred_q, y_qry[i][0]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct
        loss_q = losses_q[-1] / task_num
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        
        return corrects, losses_q[-1], pre_label, true_label

def main():
    pass

if __name__ == '__main__':
    main()
