import torch

class RowAdam(torch.optim.Adam):
    """torch.optim.Adam for a model whose tensors are packed as rows of parameter banks (TNModel).

    Banks get no .grad (plain Adam skips them). Each step takes the rows the training forwards gathered and their
    gradients (row_model.pop_row_grads()) and updates only those rows, with a step count per row. This is what
    torch.optim.Adam does with one parameter per row: tensors without a gradient are skipped and keep their own
    bias-correction count. The row update follows torch's foreach Adam (L2 weight decay added to the gradient,
    lerp for the first moment, scalar bias corrections in double precision).
    """
    def __init__(self, params, row_model, **kwargs):
        super().__init__(params, **kwargs)
        self.row_model = row_model
        self._group_of = {id(p): g for g in self.param_groups for p in g['params']}

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for bank, rows, grad in self.row_model.pop_row_grads():
            group = self._group_of[id(bank)]
            lr, (beta1, beta2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
            state = self.state[bank]
            if 'row_step' not in state:
                state['row_step'] = torch.zeros(bank.shape[0], dtype=torch.float32, device=bank.device)
                state['exp_avg'] = torch.zeros_like(bank)
                state['exp_avg_sq'] = torch.zeros_like(bank)
            step = state['row_step'].index_select(0, rows) + 1
            state['row_step'].index_copy_(0, rows, step)
            p = bank.index_select(0, rows)
            if wd != 0:
                grad = grad.add(p, alpha=wd)
            m = state['exp_avg'].index_select(0, rows).lerp_(grad, 1 - beta1)
            v = state['exp_avg_sq'].index_select(0, rows).mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            view = (-1,) + (1,) * (bank.dim() - 1)
            s = step.double()
            step_size = ((lr / (1 - beta1 ** s)) * -1).to(bank.dtype).view(view)
            bc2_sqrt = ((1 - beta2 ** s) ** 0.5).to(bank.dtype).view(view)
            denom = v.sqrt().div_(bc2_sqrt).add_(eps)
            p.addcmul_(m / denom, step_size)
            bank.index_copy_(0, rows, p)
            state['exp_avg'].index_copy_(0, rows, m)
            state['exp_avg_sq'].index_copy_(0, rows, v)
        return loss
