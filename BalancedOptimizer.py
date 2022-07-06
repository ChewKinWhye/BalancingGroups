import torch
class BalancedOptimizer(torch.optim.Optimizer):
    def init(self, params, lr=1e-3, eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(BalancedOptimizer, self).__init__(params, defaults)
    def step(self, closure=None):
        # Loop through each param group
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            # Loop through each parameter
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                param.add_(d_p, alpha=-self.lr)
