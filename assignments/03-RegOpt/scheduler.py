import math
from typing import List
import random


from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler.
    I am implementing cyclical learning rate.
    """

    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        step_size=400,
        base_lr=0.001,
        max_lr=0.01,
        gamma=0.999,
        warming_mode="linear",
        cooling_mode="exponential",
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.step_size = step_size
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warming_mode = warming_mode
        self.cooling_mode = cooling_mode
        self.gamma = gamma

        self.base_stage_size = step_size * 3
        self.warmup_stage_size = step_size
        self.cooling_stage_size = step_size * 2
        self.total_period = (
            self.base_stage_size + self.warmup_stage_size + self.cooling_stage_size
        )

        super().__init__(optimizer, last_epoch)

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get the learning rate.
        """
        lrs = []
        current_step = self.last_epoch % self.total_period
        if current_step < self.base_stage_size:
            lrs = [self.base_lr for element in self.optimizer.param_groups]
        elif current_step < self.base_stage_size + self.warmup_stage_size:
            if self.warming_mode == "exponential":
                lrs = [
                    self.base_lr
                    + (self.max_lr - self.base_lr)
                    * (self.gamma ** (current_step - self.base_stage_size))
                    for _ in self.optimizer.param_groups
                ]
            else:
                lrs = [
                    self.base_lr
                    + (self.max_lr - self.base_lr)
                    * abs(
                        current_step % self.warmup_stage_size / self.warmup_stage_size
                    )
                    for _ in self.optimizer.param_groups
                ]
        else:
            if self.cooling_mode == "exponential":
                lrs = [
                    self.max_lr
                    - (self.max_lr - self.base_lr)
                    * (
                        self.gamma
                        ** (
                            current_step - self.base_stage_size - self.warmup_stage_size
                        )
                    )
                    for _ in self.optimizer.param_groups
                ]
            else:
                lrs = [
                    self.max_lr
                    - (self.max_lr - self.base_lr)
                    * abs(
                        current_step % self.cooling_stage_size / self.cooling_stage_size
                    )
                    for _ in self.optimizer.param_groups
                ]

        return lrs
