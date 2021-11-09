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

"""Implments the model function for the POTR model."""


import numpy as np
import os
import sys
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import training.seq2seq_model_fn as seq2seq_model_fn
import models.PoseTransformer as PoseTransformer
import models.PoseEncoderDecoder as PoseEncoderDecoder
import data.H36MDataset as H36MDataset
import data.H36MDataset_v2 as H36MDataset_v2
import data.AMASSDataset as AMASSDataset
import data.NTURGDDataset as NTURGDDataset
import utils.utils as utils
import radam.radam as radam

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_WEIGHT_DECAY = 0.00001
_NSEEDS = 8

class POTRModelFn(seq2seq_model_fn.ModelFn):
  def __init__(self,
               params,
               train_dataset_fn,
               eval_dataset_fn,
               pose_encoder_fn=None,
               pose_decoder_fn=None):
    super(POTRModelFn, self).__init__(
      params, train_dataset_fn, eval_dataset_fn, pose_encoder_fn, pose_decoder_fn)
    self._loss_fn = self.layerwise_loss_fn

  def smooth_l1(self, decoder_pred, decoder_gt):
    l1loss = nn.SmoothL1Loss(reduction='mean')
    return l1loss(decoder_pred, decoder_gt)

  def loss_l1(self, decoder_pred, decoder_gt):
    return nn.L1Loss(reduction='mean')(decoder_pred, decoder_gt)

  def loss_activity(self, logits, class_gt):                                     
    """Computes entropy loss from logits between predictions and class."""
    return nn.functional.cross_entropy(logits, class_gt, reduction='mean')

  def compute_class_loss(self, class_logits, class_gt):
    """Computes the class loss for each of the decoder layers predictions or memory."""
    class_loss = 0.0
    for l in range(len(class_logits)):
      class_loss += self.loss_activity(class_logits[l], class_gt)

    return class_loss/len(class_logits)

  def select_loss_fn(self):
    if self._params['loss_fn'] == 'mse':
      return self.loss_mse
    elif self._params['loss_fn'] == 'smoothl1':
      return self.smooth_l1
    elif self._params['loss_fn'] == 'l1':
      return self.loss_l1
    else:
      raise ValueError('Unknown loss name {}.'.format(self._params['loss_fn']))

  def layerwise_loss_fn(self, decoder_pred, decoder_gt, class_logits=None, class_gt=None):
    """Computes layerwise loss between predictions and ground truth."""
    pose_loss = 0.0
    loss_fn = self.select_loss_fn()

    for l in range(len(decoder_pred)):
      pose_loss += loss_fn(decoder_pred[l], decoder_gt)

    pose_loss = pose_loss/len(decoder_pred)
    if class_logits is not None:
      return pose_loss, self.compute_class_loss(class_logits, class_gt)

    return pose_loss, None

  def init_model(self, pose_encoder_fn=None, pose_decoder_fn=None):
    self._model = PoseTransformer.model_factory(
        self._params, 
        pose_encoder_fn, 
        pose_decoder_fn
    )

  def select_optimizer(self):
    optimizer = optim.AdamW(
        self._model.parameters(), lr=self._params['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=_WEIGHT_DECAY
    )

    return optimizer


def dataset_factory(params):
  if params['dataset'] == 'h36m_v2':
    return H36MDataset_v2.dataset_factory(params)
  elif params['dataset'] == 'ntu_rgbd':
    return NTURGDDataset.dataset_factory(params)
  else:
    raise ValueError('Unknown dataset {}'.format(params['dataset']))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_prefix', type=str, default='')
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--data_path', type=str)
  parser.add_argument('--learning_rate', type=float, default=1e-5)
  parser.add_argument('--max_epochs', type=int, default=3000)
  parser.add_argument('--steps_per_epoch', type=int, default=200)
  parser.add_argument('--action', nargs='*', type=str, default=None)
  parser.add_argument('--use_one_hot',  action='store_true')
  parser.add_argument('--init_fn', type=str, default='xavier_init')
  parser.add_argument('--include_last_obs', action='store_true')
  # pose transformers related parameters
  parser.add_argument('--model_dim', type=int, default=256)
  parser.add_argument('--num_encoder_layers', type=int, default=4)
  parser.add_argument('--num_decoder_layers', type=int, default=4)
  parser.add_argument('--num_heads', type=int, default=4)
  parser.add_argument('--dim_ffn', type=int, default=2048)
  parser.add_argument('--dropout', type=float, default=0.1)
  parser.add_argument('--source_seq_len', type=int, default=50)                  
  parser.add_argument('--target_seq_len', type=int, default=25)
  parser.add_argument('--max_gradient_norm', type=float, default=0.1)
  parser.add_argument('--lr_step_size',type=int, default=400)
  parser.add_argument('--learning_rate_fn',type=str, default='step')
  parser.add_argument('--warmup_epochs', type=int, default=100)
  parser.add_argument('--pose_format', type=str, default='expmap')
  parser.add_argument('--remove_low_std', action='store_true')
  parser.add_argument('--remove_global_trans', action='store_true')
  parser.add_argument('--loss_fn', type=str, default='l1')
  parser.add_argument('--pad_decoder_inputs', action='store_true')
  parser.add_argument('--pad_decoder_inputs_mean', action='store_true')
  parser.add_argument('--use_wao_amass_joints', action='store_true')
  parser.add_argument('--non_autoregressive', action='store_true')
  parser.add_argument('--pre_normalization', action='store_true')
  parser.add_argument('--use_query_embedding', action='store_true')
  parser.add_argument('--predict_activity', action='store_true')
  parser.add_argument('--use_memory', action='store_true')
  parser.add_argument('--query_selection',action='store_true')
  parser.add_argument('--activity_weight', type=float, default=1.0)
  parser.add_argument('--pose_embedding_type', type=str, default='simple')
  parser.add_argument('--encoder_ckpt', type=str, default=None)
  parser.add_argument('--dataset', type=str, default='h36m_v2')
  parser.add_argument('--skip_rate', type=int, default=5)
  parser.add_argument('--eval_num_seeds', type=int, default=_NSEEDS)
  parser.add_argument('--copy_method', type=str, default=None)
  parser.add_argument('--finetuning_ckpt', type=str, default=None)
  parser.add_argument('--pos_enc_alpha', type=float, default=1)
  parser.add_argument('--pos_enc_beta', type=float, default=10000)
  args = parser.parse_args()
  
  params = vars(args)
  train_dataset_fn, eval_dataset_fn = dataset_factory(params)

  params['input_dim'] = train_dataset_fn.dataset._data_dim
  params['pose_dim'] = train_dataset_fn.dataset._pose_dim
  pose_encoder_fn, pose_decoder_fn = \
      PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)

  for k,v in params.items():
    print('[INFO] (POTRFn@main) {}: {}'.format(k, v))

  utils.create_dir_tree(params['model_prefix'])

  config_path = os.path.join(params['model_prefix'], 'config', 'config.json')        
  with open(config_path, 'w') as file_:
    json.dump(params, file_, indent=4)

  model_fn = POTRModelFn(
      params, train_dataset_fn, 
      eval_dataset_fn, 
      pose_encoder_fn, pose_decoder_fn
  )
  model_fn.train()



