###############################################################################
# Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive 
# Transformers
# 
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Angel Martinez <angel.martinez@idiap.ch>,
# 
# This file is part of 
# POTR: Human Motion Prediction with Non-Autoregressive Transformers
# 
# POTR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# POTR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with POTR. If not, see <http://www.gnu.org/licenses/>.
###############################################################################

"""Warm up scheduler implementation.

Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
Adapted from https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
"""

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
  """ Gradually warm-up(increasing) learning rate in optimizer."""
  def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
    """Constructor.

    Args:
      optimizer (Optimizer): Wrapped optimizer.
      multiplier: target learning rate = base lr * multiplier if 
        multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with 
        the base_lr.
      total_epoch: target learning rate is reached at total_epoch, gradually
      after_scheduler: after target_epoch, use this scheduler 
          (eg. ReduceLROnPlateau)
    """
    self.multiplier = multiplier
    if self.multiplier < 1.:
        raise ValueError('multiplier should be greater than or equal to 1.')
    self.total_epoch = total_epoch
    self.after_scheduler = after_scheduler
    self.finished = False
    super(GradualWarmupScheduler, self).__init__(optimizer)

  def get_lr(self):
    if self.last_epoch > self.total_epoch:
      if self.after_scheduler:
        if not self.finished:
          self.after_scheduler.base_lrs = [
              base_lr * self.multiplier for base_lr in self.base_lrs]
          self.finished = True
        return self.after_scheduler.get_last_lr()
      return [base_lr * self.multiplier for base_lr in self.base_lrs]

    if self.multiplier == 1.0:
      return [base_lr * (float(self.last_epoch) / self.total_epoch) 
              for base_lr in self.base_lrs]
    else:
      return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) 
              for base_lr in self.base_lrs]

  def step_ReduceLROnPlateau(self, metrics, epoch=None):
    if epoch is None:
      epoch = self.last_epoch + 1
    # ReduceLROnPlateau is called at the end of epoch, whereas others 
    # are called at beginning
    self.last_epoch = epoch if epoch != 0 else 1  
    if self.last_epoch <= self.total_epoch:
      warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) 
                  for base_lr in self.base_lrs]
      for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
        param_group['lr'] = lr
    else:
      if epoch is None:
        self.after_scheduler.step(metrics, None)
      else:
        self.after_scheduler.step(metrics, epoch - self.total_epoch)

  def step(self, epoch=None, metrics=None):
    if type(self.after_scheduler) != ReduceLROnPlateau:
      if self.finished and self.after_scheduler:
        if epoch is None:
          self.after_scheduler.step(None)
        else:
          self.after_scheduler.step(epoch - self.total_epoch)
        self._last_lr = self.after_scheduler.get_last_lr()
      else:
        return super(GradualWarmupScheduler, self).step(epoch)
    else:
      self.step_ReduceLROnPlateau(metrics, epoch)
