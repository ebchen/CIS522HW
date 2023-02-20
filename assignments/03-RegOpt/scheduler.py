import math
from typing import List


from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler.
    I am implementing cyclical learning rate.
    """

    def __init__(
        self,
        optimizer,
        base_lr=0.001,
        max_lr=0.01,
        step_size=2000,
        gamma=1.0,
        last_epoch=-1,
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.cycle = 0
        self.last_lr = 0
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get the learning rate list.
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        self.cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * self.cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * (
            self.gamma**self.last_epoch
        )
        self.last_lr = lr
        return [lr for _ in self.optimizer.param_groups]
        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]
