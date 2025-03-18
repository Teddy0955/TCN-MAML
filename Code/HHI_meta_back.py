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


class HHI_Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(HHI_Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        #self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.new_lr = args.meta_lr * args.focus_lr
        self.net = TCN_Learner(config)
    
        if(args.meta_optim == 'AdamW'):
            self.meta_optim = optim.AdamW(self.net.parameters(), lr=self.meta_lr)
        elif(args.meta_optim == 'Adam'):
            self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
       


        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.meta_optim, lr_lambda = lambda epoch: args.lr_decay ** int(epoch))

        self.test_acc = -1

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

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

        return total_norm/counter
    
    def focal_loss(self,predictions, targets, alpha, gamma):

        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets)

        # Compute the focal weights
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_weights = (alpha * (1 - pt) ** gamma)

        # Apply the focal weights to the cross-entropy loss
        focal_loss = focal_weights * ce_loss

        return focal_loss

    def forward(self, x_spt, y_spt, x_qry, y_qry, dropout_training = True):
        """

        :param x_spt:   [batch, support_way, TR_pairs, timestamp, channel]
        :param y_spt:   [batch, support_way]
        :param x_qry:   [batch, query_way, TR_pairs, timestamp, channel]
        :param y_qry:   [batch, query_way]
        :return:
        """
        task_num = x_spt.shape[0]
        
        
        pre_label = []
        true_label = []

        # losses_q[i] is the loss on step i
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]
        

        for i in range(task_num):
            is_target_domain = False
            qry_label = (y_qry[i]%113).to(int) + (y_qry[i]/113).to(int)
            # 1. run the i-th task and compute loss for k=0
            loss = 0
            for idx in range(x_spt[i].shape[0]): # run all sample in support set
                logits = self.net(x_spt[i][idx], vars=None,
                                    bn_training=True, dropout_training = dropout_training).to(torch.float)
                #---------Ｏｎｌｙ ＣＥ
                #loss += F.cross_entropy(logits, y_spt[i][idx])
                #---------Ｏｎｌｙ ＦＯＣＡＬ　ＬＯＳＳ
                loss += self.focal_loss(logits, y_spt[i][idx],self.alpha, self.gamma)

            loss /= x_spt[i].shape[0] # !!!new
            #print("----------LOSS=",loss)
            #print("loss  = ", loss)
            grad = torch.autograd.grad(loss, self.net.parameters(),allow_unused=True)
            
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(
                    x_qry[i][0], self.net.parameters(), bn_training=True, dropout_training = False)
                
                #loss_q = F.cross_entropy(logits_q, y_qry[i][0])              
                #loss_q = F.cross_entropy(logits_q, qry_label[0])
                loss_q = self.focal_loss(logits_q, qry_label[0],self.alpha, self.gamma)
                
                losses_q[0] += loss_q

                #pred_q = F.softmax(logits_q, dim=0).argmax(dim=0)
                pred_q = logits_q.argmax(dim=0)
                correct = torch.eq(pred_q, qry_label[0]).sum().item()
                corrects[0] = corrects[0] + correct

            #this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i][0], fast_weights, bn_training=True, dropout_training = False)

                #loss_q = F.cross_entropy(logits_q, y_qry[i][0])
                #loss_q = F.cross_entropy(logits_q, qry_label[0])
                loss_q = self.focal_loss(logits_q, qry_label[0],self.alpha, self.gamma)
                losses_q[1] += loss_q
                # [setsz]
                # pred_q = F.softmax(logits_q, dim=0).argmax(dim=0)
                pred_q = logits_q.argmax(dim=0)
                correct = torch.eq(pred_q, qry_label[0]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                loss = 0
                for idx in range(x_spt[i].shape[0]):
                    logits = self.net(x_spt[i][idx], fast_weights, bn_training=True, dropout_training = dropout_training,)
                    #---------Ｏｎｌｙ ＣＥ
                    #loss += F.cross_entropy(logits, y_spt[i][idx])
                    #---------Ｏｎｌｙ ＦＯＣＡＬ　ＬＯＳＳ
                    loss += self.focal_loss(logits, y_spt[i][idx],self.alpha, self.gamma)

                loss /= x_spt[i].shape[0] # !!!new
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights,allow_unused=True)

                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i][0], fast_weights, bn_training=True, dropout_training = False)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                #loss_q = F.cross_entropy(logits_q, qry_label[0])
                loss_q = self.focal_loss(logits_q, qry_label[0],self.alpha, self.gamma)

                losses_q[k + 1] += loss_q

                ### ＮＥＷ！！！
                # if(k<self.update_step-1):
                #     losses_q[k + 1] += loss_q
                # else:
                #     if(is_target_domain):
                #         target_loss += loss_q
                #         target_cnt +=1
                #     else:
                #         losses_q[k + 1] += loss_q


                with torch.no_grad():
                    # pred_q = F.softmax(logits_q, dim=0).argmax(dim=0)
                    pred_q = logits_q.argmax(dim=0)
                    if(k==self.update_step-1):
                        pre_label.append(pred_q.item())
                        true_label.append(qry_label[0].item())
                    correct = torch.eq(pred_q, qry_label[0]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        ###　ＮＥＷ　ＬＯＳＳ
        # loss_q = losses_q[-1] / (task_num-target_cnt)
        
        self.meta_optim.zero_grad()
        loss_q.backward()

        # if(target_cnt !=0):
        #     target_loss /= target_cnt
        #     for param_group in self.meta_optim.param_groups:
        #         param_group['lr'] = self.new_lr
        #     self.meta_optim.zero_grad()
        #     target_loss.backward(retain_graph=True)

        #     for param_group in self.meta_optim.param_groups:
        #         param_group['lr'] = self.meta_lr

        self.meta_optim.step()

        return corrects, losses_q[-1], pre_label, true_label

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, dropout_training=False):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        #assert len(x_spt.shape) == 4

        task_num = x_spt.shape[0]
        querysz = x_qry.size(0)
        total_loss = 0

        pre_label = []
        true_label = []

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        for i in range(task_num):

            qry_label = (y_qry[i]%113).to(int) + (y_qry[i]/113).to(int)

            # qry_label = y_qry[i].clone()
            # if(int(qry_label)/113!=0):
            #     is_target_domain = True
            #     qry_label = (qry_label/113).to(int)
            loss = 0
            # 1. run the i-th task and compute loss for k=0
            for idx in range(x_spt[i].shape[0]):
                logits = net(x_spt[i][idx], vars=None,
                                    bn_training=True, dropout_training = dropout_training).to(torch.float)
                #---------ＣＥ
                #loss += F.cross_entropy(logits, y_spt[i][idx])
                loss += self.focal_loss(logits, y_spt[i][idx],self.alpha, self.gamma)

            loss /= x_spt[i].shape[0] # !!!new
            grad = torch.autograd.grad(loss, net.parameters(),allow_unused=True)
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
            
            for k in range(1, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                loss = 0
                for idx in range(x_spt[i].shape[0]):
                    logits = net(x_spt[i][idx], fast_weights, bn_training=True, dropout_training = dropout_training)
                    #---------ＣＥ
                    # loss += F.cross_entropy(logits, y_spt[i][idx])
                    loss += self.focal_loss(logits, y_spt[i][idx],self.alpha, self.gamma)
                loss /= x_spt[i].shape[0] # !!!new

                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights,allow_unused=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = net(x_qry[i][0], fast_weights, bn_training=True, dropout_training = dropout_training)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                #loss_q = F.cross_entropy(logits_q, qry_label[0])
                loss_q = self.focal_loss(logits_q, qry_label[0],self.alpha, self.gamma)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=0).argmax(dim=0)
                    # convert to numpy
                    if(k==self.update_step_test-1):
                        pre_label.append(pred_q.item())
                        true_label.append(qry_label[0].item())
                    # correct = torch.eq(pred_q, y_qry[i][0]).sum().item()
                    corrects[k + 1] += torch.eq(pred_q, qry_label[0]).sum().item()
            total_loss += loss_q

        del net

        #accs = np.array(corrects) / querysz

        #return accs
        return corrects, total_loss, pre_label, true_label
        
def main():
    pass


if __name__ == '__main__':
    main()
