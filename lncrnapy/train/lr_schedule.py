'''Simple learning rate schedule implementation.

Huang et al. (2022) https://nlp.seas.harvard.edu/annotated-transformer'''

import torch


class LrSchedule(torch.optim.lr_scheduler.LambdaLR):
    '''Linearly increases the learning rate for the first warmup_steps, then
    then decreases the learning rate proportionally to 1/sqrt(step_number)'''

    def __init__(self, optimizer, d_model, warmup_steps):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, lambda step: self._get_lr(step))

    def _get_lr(self, step):
        if step == 0:
            step = 1
        return (self.d_model**(-0.5) * 
                min(step**(-0.5), step * self.warmup_steps ** (-1.5)))