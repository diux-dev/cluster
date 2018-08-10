import torch
import torch.nn as nn
import numpy as np

# Our own implementation of lars
class LARS(torch.optim.Optimizer):
    # SGD https://raw.githubusercontent.com/pytorch/pytorch/master/torch/optim/sgd.py
    # η (eta) = "trust" coefficient
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, eta=0.02, eps=1e-8, lars=True, sgd_lars_ratio=0):
        self.lars = lars
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, eta=eta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.sgd_lars_ratio = sgd_lars_ratio
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        if self.sgd_lars_ratio >= 1: self.lars = False
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            eta = group['eta']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if self.lars: local_lr = eta * torch.norm(p.data) / torch.norm(d_p)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                if self.lars: 
                    ratio = (1 - local_lr) * (self.sgd_lars_ratio) + local_lr
                    p.data.add_(-min(ratio*group['lr'], group['lr']), d_p)
                else: p.data.add_(-group['lr'], d_p)

        return loss


def init_dist_weights(model):
    # https://arxiv.org/pdf/1706.02677.pdf
    # https://github.com/pytorch/examples/pull/262
    for m in model.modules():
        if isinstance(m, BasicBlock): m.bn2.weight = Parameter(torch.zeros_like(m.bn2.weight))
        if isinstance(m, Bottleneck): m.bn3.weight = Parameter(torch.zeros_like(m.bn3.weight))
        if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

from functools import partial
class MixUpDataLoader(object):
    """
    Creates a new data loader with mixup from a given dataloader.
    
    Mixup is applied between a batch and a shuffled version of itself. 
    If we use a regular beta distribution, this can create near duplicates as some lines might be 
    1 * original + 0 * shuffled while others could be 0 * original + 1 * shuffled, this is why
    there is a trick where we take the maximum of lambda and 1-lambda.
    
    Arguments:
    dl (DataLoader): the data loader to mix up
    alpha (float): value of the parameter to use in the beta distribution.
    """
    def __init__(self, dl, alpha):
        self.dl, self.alpha = dl, alpha
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for (x, y) in iter(self.dl):
            #Taking one different lambda per image speeds up training 
            lambd = np.random.beta(self.alpha, self.alpha, y.size(0))
            #Trick to avoid near duplicates
            lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
#             lambd = to_gpu(VV(lambd))
            lambd = torch.from_numpy(lambd).cuda().float()
            shuffle = torch.randperm(y.size(0))
            x = x.cuda().half()
            x1, y1 = x[shuffle], y[shuffle]
            lamd_rs = lambd.view(lambd.size(0),1,1,1).half()
            # lambd = lambd.view(lambd.size(0),1,1,1).half()
            new_x = x * lamd_rs + x1 * (1-lamd_rs)
            yield (new_x, [y, y1, lambd.half()])

class MixUpLoss(nn.Module):
    """
    Adapts the loss function to go with mixup.
    
    Since the targets aren't one-hot encoded, we use the linearity of the loss function with
    regards to the target to mix up the loss instead of one-hot encoded targets.
    
    Argument:
    crit: a loss function. It must have the parameter reduced=False to have the loss per element.
    """
    def __init__(self, crit):
        super().__init__()
        self.crit = crit()
        
    def forward(self, output, target):
        if not isinstance(target, list): return self.crit(output, target).mean()
        loss1, loss2 = self.crit(output,target[0]), self.crit(output,target[1])
        return (loss1 * target[2] + loss2 * (1-target[2])).mean()

import torch

# Filter out batch norm parameters and remove them from weight decay - gets us higher accuracy 93.2 -> 93.48
# https://arxiv.org/pdf/1807.11205.pdf
def bnwd_optim_params(model, model_params, master_params):
    bn_params, remaining_params = split_bn_params(model, model_params, master_params)
    return [{'params':bn_params,'weight_decay':0}, {'params':remaining_params}]


def split_bn_params(model, model_params, master_params):
    def get_bn_params(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): return module.parameters()
        accum = set()
        for child in module.children(): [accum.add(p) for p in get_bn_params(child)]
        return accum
    
    mod_bn_params = get_bn_params(model)
    zipped_params = list(zip(model_params, master_params))

    mas_bn_params = [p_mast for p_mod,p_mast in zipped_params if p_mod in mod_bn_params]
    mas_rem_params = [p_mast for p_mod,p_mast in zipped_params if p_mod not in mod_bn_params]
    return mas_bn_params, mas_rem_params
    