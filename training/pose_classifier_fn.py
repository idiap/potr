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

"""
    
[1] https://arxiv.org/abs/1312.6114
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter 
import torch.optim as optim 

import numpy as np
import os
import sys
import argparse
import tqdm

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import models.PoseActionClassifier as ActionClass
import data.H36MDatasetPose as H36MDataset
import utils.utils as utils

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PoseActionFn(object):
  def __init__(self, params, train_dataset, val_dataset=None):
    self._params = params
    self._train_dataset = train_dataset
    self._val_dataset = val_dataset
    self._writer = SummaryWriter(
        os.path.join(self._params['model_prefix'], 'tf_logs'))

    self.load_constants()
    self.init_model()
    self.select_optimizer()
    thisname = self.__class__.__name__
    self._lr_scheduler = utils.get_lr_fn(self._params, self._optimizer_fn)

    for k, v in self._params.items():
      print('[INFO] ({}) {}: {}'.format(thisname, k, v))

  def load_constants(self):
    self._params['use_one_hot'] = False
    self._params['parent'], self._params['offset'], \
      self._params['rot_ind'], self._params['exp_map_ind'] = \
        utils.load_constants(self._params['data_path'])

  def init_model(self):
    self._model = ActionClass.ActionClassifier(
        dim=self._params['model_dim'], 
        n_classes=len(self._params['action_subset'])
    )
    self._model.to(_DEVICE)
    n_params = filter(lambda p: p.requires_grad, self._model.parameters())
    n_params = sum([np.prod(p.size()) for p in n_params])
    print('++++++++ Total Parameters:', n_params)

  def select_optimizer(self):
    self._optimizer_fn = optim.Adam(
        self._model.parameters(), 
        lr=self._params['learning_rate']
    )

  def compute_accuracy(self, class_logits, class_gt):
    class_pred = torch.argmax(class_logits.softmax(-1), -1)                      
    accuracy = (class_pred == class_gt).float().sum()                            
    accuracy = accuracy / (class_logits.size()[0]) 
    return accuracy

  def forward_model(self, sample):
    pose_gt = sample['pose'].to(_DEVICE)
    class_gt = sample['action'].to(_DEVICE)
    class_logits = self._model(pose_gt)
    loss = nn.functional.cross_entropy(class_logits, class_gt, reduction='mean')
    accuracy = self.compute_accuracy(class_logits, class_gt)

    return loss, accuracy

  def train_one_epoch(self, epoch):
    epoch_loss, epoch_accuracy = 0, 0
    N = len(self._train_dataset)
    self._model.train()
    for i, sample in enumerate(self._train_dataset):
      self._optimizer_fn.zero_grad()

      loss, accuracy = self.forward_model(sample)

      if i%1000 == 0:
        print('[INFO] epoch: {:04d}; it: {:04d} loss: {:.4f}; acc: {:.4f}'.format(
            epoch, i, loss, accuracy))

      loss.backward()
      self._optimizer_fn.step() 

      epoch_loss += loss
      epoch_accuracy += accuracy

    return epoch_loss/N, epoch_accuracy/N

  @torch.no_grad()
  def validation(self, epoch):
    epoch_loss, epoch_accuracy = 0, 0
    N = len(self._val_dataset)
    self._model.eval()
    for i, sample in tqdm.tqdm(enumerate(self._val_dataset)):
      loss, accuracy = self.forward_model(sample)
      epoch_loss += loss
      epoch_accuracy += accuracy

    return epoch_loss/N, epoch_accuracy/N

  def train(self):
    thisname = self.__class__.__name__
    self._params['learning_rate'] = self._lr_scheduler.get_last_lr()[0]
    for e in range(self._params['max_epochs']):
      self._model.train()
      epoch_loss, epoch_accuracy = self.train_one_epoch(e)
      val_loss, val_accuracy = self.validation(e)

      # save models
      model_path = os.path.join(
          self._params['model_prefix'], 'models', 'ckpt_epoch_%04d.pt'%e)
      torch.save(self._model.state_dict(), model_path)

      # verbose and write the scalars
      print('[INFO] Epoch: {:04d}; epoch_loss: {:.4f}; epoch_accuracy: {:.4f}; val_loss: {:.4f}; val_accuracy: {:.4f}; lr: {:2.2e}'.format(
          e, epoch_loss, epoch_accuracy, val_loss, val_accuracy, self._params['learning_rate']))
      self._writer.add_scalars(
        'loss/loss', {'train': epoch_loss, 'val': val_loss}, e)
      self._writer.add_scalars(
        'accurracy/accurracy', {'train': epoch_accuracy, 'val': val_accuracy}, e)
      self._writer.add_scalar(                                                                    
         'learning_rate/lr', self._params['learning_rate'], e)

      self._lr_scheduler.step(e)
      self._params['learning_rate'] = self._lr_scheduler.get_last_lr()[0]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default=None)
  parser.add_argument('--action', type=str, nargs='*', default=None)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--pose_format', type=str, default='expmap')
  parser.add_argument('--remove_low_std', action='store_true')
  parser.add_argument('--model_dim', type=int, default=128)
  parser.add_argument('--max_epochs', type=int, default=500)
  parser.add_argument('--model_prefix', type=str, default=None)
  parser.add_argument('--learning_rate', type=float, default=1e-3)
  parser.add_argument('--learning_rate_fn', type=str, default='linear')
  args = parser.parse_args()

  params = vars(args)                                                            
  if 'all' in args.action:                                                       
    args.action = H36MDataset._ALL_ACTIONS

  params['action_subset'] = args.action

  dataset_t = H36MDataset.H36MDataset(params, mode='train')                                  
  dataset_v = H36MDataset.H36MDataset(
      params, mode='eval', norm_stats=dataset_t._norm_stats)

  train_dataset_fn= torch.utils.data.DataLoader(
      dataset_t,
      batch_size=params['batch_size'],
      shuffle=True,
      num_workers=4,
      collate_fn=H36MDataset.collate_fn,
      drop_last=True
  )

  val_dataset_fn = torch.utils.data.DataLoader(
      dataset_v,
      batch_size=1,
      shuffle=True,
      num_workers=1,
      collate_fn=H36MDataset.collate_fn,
      drop_last=True
  )

  params['input_dim'] = train_dataset_fn.dataset._data_dim                       
  params['pose_dim'] = train_dataset_fn.dataset._pose_dim 

  vae_trainer = PoseActionFn(params, train_dataset_fn, val_dataset_fn)

  vae_trainer.train()


