import torch
import numpy as np
from scipy.stats import norm
from torch.optim.lr_scheduler import _LRScheduler


def PIT(transformer_pred, y_true):
    mean = transformer_pred[:, :, 0].cpu().detach().numpy()
    var = torch.nn.functional.softplus(transformer_pred[:, :, 1])
    std = np.sqrt(var.cpu().detach().numpy())

    U = norm.cdf(
        y_true.cpu().detach().numpy(),
        loc=mean,
        scale=std,
    )
    return U


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_warmup_steps, after_scheduler=None):
        self.total_warmup_steps = total_warmup_steps
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.total_warmup_steps:
            return [
                base_lr * float(self.last_epoch) / self.total_warmup_steps
                for base_lr in self.base_lrs
            ]
        if self.after_scheduler:
            if not self.finished_warmup:
                self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                self.finished_warmup = True
            return self.after_scheduler.get_last_lr()
        return self.base_lrs

    def step(self, epoch=None):
        if self.finished_warmup and self.after_scheduler:
            self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
