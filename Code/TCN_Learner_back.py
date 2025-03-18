import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils import weight_norm



class TCN_Learner(nn.Module):
    """

    """

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        """
        super(TCN_Learner, self).__init__()

        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name == 'conv1d':
                # (out_channels, input_channels, kernel_size)
                
                w = nn.Parameter(torch.ones(param[1], param[0], param[2])).cuda()
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                #torch.nn.init.uniform_(w,a=0,b=0.01)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1]).to(torch.float).cuda()))

            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(param[0],param[1])).cuda()
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                #torch.nn.init.uniform_(w,a=0,b=1) #init w from 0 to 1
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0]).to(torch.float).cuda()))

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]),requires_grad=True).cuda()
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0],requires_grad=True).cuda()))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])                
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool1d', 'dropout',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid','softmax',
                          'pick_time', 'save_embedding']:
                continue
            else:
                raise NotImplementedError

    def forward(self, x, vars=None, bn_training=True, dropout_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
           # print("-----------Now ", name)
            if name == 'conv1d':
                w, b = vars[idx], vars[idx + 1]
                # padding for implementing the causal and dilation mechanism of TCN
                #print("************x=",x.shape)
                # print("---------------Weight--------------")
                # print(w)
                # print("----------------Bias--------------")
                # print(b)
                # print("-----------------------------------")

                # new_w  = self.normalize_wieght(w)
                if(x.shape[-1] != 375):
                    x = x.permute(0,2,1)
                x = F.pad(x, (int(param[3]*(param[2]-1)), 0), mode='constant')
                x = F.conv1d(x, w, b, stride=1,
                             padding='valid', dilation=param[3])
                idx += 2
                #print(name, param, '\tout:', x.shape
            elif name == 'dropout':
                x = F.dropout(x, param[0], training=dropout_training)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                # print("---------------Weight--------------")
                # print(w)
                # print("----------------Bias--------------")
                # print(b)
                # print("-----------------------------------")
                # input()
                # if(x.shape[2]!=375):
                #     x=x.permute(0,2,1)
                # x = x[:,:,-1]
                #x = torch.flatten(x, start_dim=0, end_dim=-1)
                # print(x[:10])
                x = F.linear(x, w, b)
                idx += 2
                # print(name, param, '\tout:', x.shape, '\t\t\tforward:', idx, x.norm().item())
                # print(x[:10])
                # print('forward:', idx, x.norm().item())
            elif name == 'avg_pool1d':
                x = torch.transpose(x,0,1)
                #print(name, param, '\tinput:', x.shape)

                x = F.avg_pool1d(x, kernel_size=param[0])
                x = x.squeeze(-1)
                #print(name, param, '\tout:', x.shape)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean.cuda(), running_var.cuda(),
                                 weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name =='softmax':
                x = F.softmax(x,dim=-1)
            elif name == 'flatten':
                x = torch.flatten(x,start_dim=0, end_dim=-1)
            elif name == 'pick_time':
                x = x[:,:,param]
            elif name == 'save_embedding':
                torch.save(x,)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def normalize_wieght(self, w, eps=1e-12):
        # eps is used for preventing division by zero
        # new_w = nn.Parameter(w.data)
        new_w = w.data
        nor_w = torch.sqrt(torch.sum(new_w ** 2)) + eps
        return new_w/nor_w 


    def _norm(self,p, dim):
        if dim is None:
            return p.norm()
        elif dim == 0:
            output_size = (p.size(0),) + (1,) * (p.dim() - 1)
            return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
        elif dim == p.dim() - 1:
            output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
            return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
        else:
            return self._norm(p.transpose(0, dim), 0).transpose(0, dim)

    # def compute_weight(self, model, name):
    #     g = getattr(model, name + '_g')
    #     v = getattr(model, name + '_v')
    #     return v * (g / _norm(v, self.dim))

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
