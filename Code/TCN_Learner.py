import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils import weight_norm


class TCN_Learner(nn.Module):
    def __init__(self, config):
        super(TCN_Learner, self).__init__()
        
        self.config = config
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name == 'conv1d':
                # Create a torch.nn.Parameter object for weight 'w'
                w = nn.Parameter(torch.ones(param[1], param[0], param[2]).cuda())
                # Initialize the weight using kaiming_normal_ method
                torch.nn.init.kaiming_normal_(w)
                # Append the weight to self.vars
                self.vars.append(w)

                # Create a torch.nn.Parameter object for bias 'b'
                b = nn.Parameter(torch.zeros(param[1]).to(torch.float).cuda())
                # Append the bias to self.vars
                self.vars.append(b)

            elif name == 'linear':
                # Create a torch.nn.Parameter object for weight 'w'
                w = nn.Parameter(torch.ones(param[0], param[1]).cuda())
                # Initialize the weight using kaiming_normal_ method
                torch.nn.init.kaiming_normal_(w)
                # Append the weight to self.vars
                self.vars.append(w)

                # Create a torch.nn.Parameter object for bias 'b'
                b = nn.Parameter(torch.zeros(param[0]).to(torch.float).cuda())
                # Append the bias to self.vars
                self.vars.append(b)

            elif name == 'bn':
                # Create a torch.nn.Parameter object for weight 'w'
                w = nn.Parameter(torch.ones(param[0], requires_grad=True).cuda())
                # Append the weight to self.vars
                self.vars.append(w)

                # Create a torch.nn.Parameter object for bias 'b'
                b = nn.Parameter(torch.zeros(param[0], requires_grad=True).cuda())
                # Append the bias to self.vars
                self.vars.append(b)

                # Create running_mean and running_var parameters with requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0], requires_grad=False))
                running_var = nn.Parameter(torch.ones(param[0], requires_grad=False))
                self.vars_bn.extend([running_mean, running_var])
                
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool1d', 'dropout',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'softmax',
                          'pick_time', 'save_embedding']:
                continue
            else:
                raise NotImplementedError

    def forward(self, x, vars=None, bn_training=True, dropout_training=True):
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name == 'conv1d':
                w, b = vars[idx], vars[idx + 1]
                if x.shape[-1] != 375:
                    x = x.permute(0, 2, 1)
                x = F.pad(x, (int(param[3] * (param[2] - 1)), 0), mode='constant')
                x = F.conv1d(x, w, b, stride=1, padding='valid', dilation=param[3])
                idx += 2
                
            elif name == 'dropout':
                x = F.dropout(x, param[0], training=dropout_training)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name == 'avg_pool1d':
                x = torch.transpose(x, 0, 1)
                x = F.avg_pool1d(x, kernel_size=param[0])
                x = x.squeeze(-1)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean.cuda(), running_var.cuda(), weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'softmax':
                x = F.softmax(x, dim=-1)
            elif name == 'flatten':
                x = torch.flatten(x, start_dim=0, end_dim=-1)
            elif name == 'pick_time':
                x = x[:, :, param]
            elif name == 'save_embedding':
                # Save embedding logic here
                pass
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            else:
                raise NotImplementedError

        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars

