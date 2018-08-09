import torch

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