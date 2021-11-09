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

"""Implements a model function estimator for training, evaluation and predict.

Take and adapted from the code presented in [4]

[1] https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
[2] https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
[3] https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-247769197
[4] https://arxiv.org/pdf/1705.02445.pdf
"""

import sys
import numpy as np
import json
import sys
import os
import argparse
import time
from abc import abstractmethod
import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

sys.path.append('../')


import utils.utils as utils
import utils.WarmUpScheduler as warm_up_scheduler
import data.H36MDataset as h36mdataset_fn
import visualize.viz as viz
import models.seq2seq_model as seq2seq_model

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# min threshold for mean average precision in metters
# Set to 10 cm
_MAP_TRESH = 0.10

class ModelFn(object):
  """Implements the model functionalities: training, evaliation and prediction."""

  def __init__(
      self,
      params,
      train_dataset_fn=None,
      eval_dataset_fn=None,
      pose_encoder_fn=None,
      pose_decoder_fn=None):
    """Initialization of model function."""
    self._params = params
    self._train_dataset_fn = train_dataset_fn
    self._eval_dataset_fn = eval_dataset_fn
    self._visualize = False
    thisname = self.__class__.__name__
    self._norm_stats = train_dataset_fn.dataset._norm_stats
    self._ms_range = [80, 160, 320, 400, 560, 1000]
    self.init_model(pose_encoder_fn, pose_decoder_fn)
    self._loss_fn = self.loss_mse
    self._model.to(_DEVICE)
    self._optimizer_fn = self.select_optimizer()
    self.select_lr_fn()
    self.finetune_init()

    self._lr_db_curve = []

    lr_type = 'stepwise' if self._params['learning_rate_fn'] == 'beatles' \
        else 'epochwise'
    self._params['lr_schedule_type'] = lr_type

    self.evaluate_fn = self.evaluate_h36m 
    if self._params['dataset'] == 'ntu_rgbd':
      self.evaluate_fn = self.evaluate_nturgbd
    elif self._params['dataset'] == 'amass':
      self.evaluate_fn = self.evaluate_amass

    self._writer = SummaryWriter(
        os.path.join(self._params['model_prefix'], 'tf_logs'))
    self._time_range_eval = []

    m_params = filter(lambda p: p.requires_grad, self._model.parameters())
    nparams = sum([np.prod(p.size()) for p in m_params])             
    print('[INFO] ({}) This module has {} parameters!'.format(thisname, nparams))
    print('[INFO] ({}) Intializing ModelFn with params'.format(thisname))
    for k,v in self._params.items():
      print('[INFO] ({}) {}: {}'.format(thisname, k, v))

  def finetune_init(self):
    if self._params['finetuning_ckpt'] is not None:
      print('[INFO] (finetune_model) Finetuning from:', 
          self._params['finetuning_ckpt'])
      self._model.load_state_dict(torch.load(
          self._params['finetuning_ckpt'], map_location=_DEVICE)
      )

  def select_lr_fn(self):
    """Calls the selection of learning rate function."""
    self._lr_scheduler = self.get_lr_fn()
    lr_fn = self._params['learning_rate_fn']
    if self._params['warmup_epochs'] > 0 and lr_fn != 'beatles':
      self._lr_scheduler = warm_up_scheduler.GradualWarmupScheduler(
          self._optimizer_fn, multiplier=1, 
          total_epoch=self._params['warmup_epochs'], 
          after_scheduler=self._lr_scheduler
      )

  def get_lr_fn(self):
    """Creates the function to be used to generate the learning rate."""
    if self._params['learning_rate_fn'] == 'step':
      return torch.optim.lr_scheduler.StepLR(
        self._optimizer_fn, step_size=self._params['lr_step_size'], gamma=0.1
      )
    elif self._params['learning_rate_fn'] == 'exponential':
      return torch.optim.lr_scheduler.ExponentialLR(
        self._optimizer_fn, gamma=0.95
      )
    elif self._params['learning_rate_fn'] == 'linear':
      # sets learning rate by multipliying initial learning rate times a function
      lr0, T = self._params['learning_rate'], self._params['max_epochs']
      lrT = lr0*0.5
      m = (lrT - 1) / T
      lambda_fn =  lambda epoch: m*epoch + 1.0
      return torch.optim.lr_scheduler.LambdaLR(
        self._optimizer_fn, lr_lambda=lambda_fn
      )
    elif self._params['learning_rate_fn'] == 'beatles':
      # D^(-0.5)*min(i^(-0.5), i*warmup_steps^(-1.5))
      D = float(self._params['model_dim'])
      warmup = self._params['warmup_epochs']
      lambda_fn = lambda e: (D**(-0.5))*min((e+1.0)**(-0.5), (e+1.0)*warmup**(-1.5))
      return torch.optim.lr_scheduler.LambdaLR(
        self._optimizer_fn, lr_lambda=lambda_fn
      )
    else:
      raise ValueError('Unknown learning rate function: {}'.format(
          self._params['learning_rate_fn']))

  @abstractmethod
  def init_model(self, pose_encoder_fn, pose_decoder_fn):
    pass

  @abstractmethod
  def select_optimizer(self):
    pass

  def loss_mse(self, decoder_pred, decoder_gt):
    """Computes the L2 loss between predictions and ground truth."""
    step_loss = (decoder_pred - decoder_gt)**2
    step_loss = step_loss.mean()

    return step_loss

  @abstractmethod
  def compute_loss(self, inputs=None, target=None, preds=None, class_logits=None, class_gt=None):
    return self._loss_fn(preds, target, class_logits, class_gt)

  def print_logs(self, step_loss, current_step, pose_loss, activity_loss, selection_loss):
    selection_logs = ''
    if self._params['query_selection']:
      selection_logs = 'selection loss {:.4f}'.format(selection_loss)
    if self._params['predict_activity']:
      print("[INFO] global {:06d}; step {:04d}; pose_loss {:4f} - class_loss {:4f}; step_loss: {:.4f}; lr: {:.2e} {:s}".\
          format(self._global_step, current_step, pose_loss, activity_loss, 
                step_loss, self._params['learning_rate'], selection_logs) 
      )
    else:
      print("[INFO] global {3:06d}; step {0:04d}; step_loss: {1:.4f}; lr: {2:.2e} {4:s}".\
          format(current_step, step_loss, self._params['learning_rate'], 
              self._global_step, selection_logs)
    )

  def compute_selection_loss(self, inputs, target, cols_softmax=False):
    """Compute the query entry selection loss.

    Args:
      inputs: [batch_size, src_len, tgt_len]
      target: [batch_size, src_len, tgt_len]
    """
    axis_ = 2 if cols_softmax else 1
    target = F.softmax(-target, dim=axis_)
    return torch.nn.MSELoss(reduction='mean')(inputs, target)

  def train_one_epoch(self, epoch):
    """Trains for a number of steps before evaluation."""
    epoch_loss = 0
    act_loss = 0
    sel_loss = 0
    N = len(self._train_dataset_fn)
    for current_step, sample in enumerate(self._train_dataset_fn):
      self._optimizer_fn.zero_grad()
      for k in sample.keys():
        if k == 'actions' or k == 'decoder_outputs_euler' or k=='action_str':
          continue
        sample[k] = sample[k].to(_DEVICE)

      decoder_pred = self._model(
          sample['encoder_inputs'], sample['decoder_inputs'])

      selection_loss = 0
      if self._params['query_selection']:
        prob_mat = decoder_pred[-1][-1]
        selection_loss = self.compute_selection_loss(
            inputs=prob_mat, 
            target=sample['src_tgt_distance']
        )
        sel_loss += selection_loss

      pred_class, gt_class = None, None
      if self._params['predict_activity']:
        gt_class = sample['action_ids']  # one label for the sequence
        pred_class = decoder_pred[1]

      pose_loss, activity_loss = self.compute_loss(
          inputs=sample['encoder_inputs'],
          target=sample['decoder_outputs'],
          preds=decoder_pred[0],
          class_logits=pred_class,
          class_gt=gt_class
      )

      step_loss = pose_loss + selection_loss
      if self._params['predict_activity']:
        step_loss += self._params['activity_weight']*activity_loss
        act_loss += activity_loss
      epoch_loss += step_loss.item()

      step_loss.backward()
      if self._params['max_gradient_norm'] is not None:
        torch.nn.utils.clip_grad_norm_(
          self._model.parameters(), self._params['max_gradient_norm'])
      self._optimizer_fn.step()

      if current_step % 10 == 0:
        step_loss = step_loss.cpu().data.numpy()
        self.print_logs(step_loss, current_step, pose_loss, activity_loss, 
            selection_loss)

      self.update_learning_rate(self._global_step, mode='stepwise')
      self._global_step += 1

    if self._params['query_selection']:
      self._scalars['train_selectioin_loss'] = sel_loss/N

    if self._params['predict_activity']:
      return epoch_loss/N, act_loss/N

    return epoch_loss/N

  def train(self):
    """Main training loop."""
    self._params['learning_rate'] = self._lr_scheduler.get_last_lr()[0]
    self._global_step = 1
    thisname = self.__class__.__name__
    for e in range(self._params['max_epochs']):
      self._scalars = {}
      self._model.train()
      start_time = time.time()
      epoch_loss = self.train_one_epoch(e)

      act_log = ''
      if self._params['predict_activity']:
        act_loss = epoch_loss[1]
        epoch_loss = epoch_loss[0]
        act_log = '; activity_loss: {}'.format(act_loss)
        self._scalars['act_loss_train'] = act_loss

      self._scalars['epoch_loss'] = epoch_loss
      print("epoch {0:04d}; epoch_loss: {1:.4f}".format(e, epoch_loss)+act_log)
      self.flush_extras(e, 'train')

      _time = time.time() - start_time
      self._model.eval()
      eval_loss = self.evaluate_fn(e, _time)

      act_log = ''
      if self._params['predict_activity']:
        self._scalars['act_loss_eval'] = eval_loss[1]
        self._scalars['accuracy'] = eval_loss[2]
        act_log = '; act_eval_loss {}; accuracy {}'.format(eval_loss[1], eval_loss[2]) 
        eval_loss = eval_loss[0]

      self._scalars['eval_loss'] = eval_loss
      print("[INFO] ({}) Epoch {:04d}; eval_loss: {:.4f}; lr: {:.2e}".format(
          thisname, e, eval_loss, self._params['learning_rate'])+act_log)

      self.write_summary(e)
      model_path = os.path.join(
          self._params['model_prefix'], 'models', 'ckpt_epoch_%04d.pt'%e)
      if (e+1)%100 == 0:
        torch.save(self._model.state_dict(), model_path)

      self.update_learning_rate(e, mode='epochwise')
      self.flush_extras(e, 'eval')
    # save the last one
    model_path = os.path.join(
        self._params['model_prefix'], 'models', 'ckpt_epoch_%04d.pt'%e)
    torch.save(self._model.state_dict(). model_path)
    # self.flush_curves()

  def write_summary(self, epoch):
    # for action_, ms_errors_ in ms_eval_loss.items():
    self._writer.add_scalars(
       'loss/recon_loss', 
        {'train':self._scalars['epoch_loss'], 'eval': self._scalars['eval_loss']}, 
        epoch
    )

    # write scalars for H36M dataset prediction style
    action_ = self._train_dataset_fn.dataset._monitor_action
    if 'ms_eval_loss' in self._scalars.keys():
      range_len = len(self._scalars['ms_eval_loss'][action_])
      # range_len = len(self._ms_range)
      ms_dict = {str(self._ms_range[i]): self._scalars['ms_eval_loss'][action_][i] 
                 for i in range(range_len)}
      ms_e = np.concatenate([np.array(v).reshape(1,range_len) 
                            for k,v in self._scalars['ms_eval_loss'].items()], axis=0)
      self._writer.add_scalars('ms_loss/eval-'+action_, ms_dict, epoch)

      ms_e = np.mean(ms_e, axis=0)  # (n_actions)
      self._time_range_eval.append(np.expand_dims(ms_e, axis=0)) # (1, n_actions)
      all_ms = {str(self._ms_range[i]): ms_e[i] for i in range(len(ms_e))}
      self._writer.add_scalars('ms_loss/eval-all', all_ms, epoch)

      self._writer.add_scalar('MSRE/msre_eval', self._scalars['msre'], epoch)
      self._writer.add_scalars('time_range/eval', 
          {'short-term':np.mean(ms_e[:4]), 'long-term':np.mean(ms_e)}, epoch)

    if self._params['predict_activity']:
      self._writer.add_scalars(
          'loss/class_loss', 
          {'train': self._scalars['act_loss_train'], 'eval': self._scalars['act_loss_eval']}, 
          epoch
      )
      self._writer.add_scalar('class/accuracy', self._scalars['accuracy'], epoch)

    if self._params['query_selection']:
      self._writer.add_scalars(
          'selection/query_selection', 
          {'eval': self._scalars['eval_selection_loss'], 
           'train': self._scalars['train_selectioin_loss']},
          epoch
      )

    if 'mAP' in self._scalars.keys():
      self._writer.add_scalar('mAP/mAP', self._scalars['mAP'], epoch)

    if 'MPJPE' in self._scalars.keys():
      self._writer.add_scalar('MPJPE/MPJPE', self._scalars['MPJPE'], epoch)

  def print_range_summary(self, action, mean_mean_errors):
    mean_eval_error = []
    # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
    print("{0: <16} |".format(action), end="")
    for ms in [1,3,7,9,13,24]:
      if self._params['target_seq_len'] >= ms + 1:
        print(" {0:.3f} |".format(mean_mean_errors[ms]), end="")
        mean_eval_error.append(mean_mean_errors[ms])
      else:
        print("   n/a |", end="")
    print()
    return mean_eval_error

  def print_table_header(self):
    print()
    print("{0: <16} |".format("milliseconds"), end="")
    for ms in [80, 160, 320, 400, 560, 1000]:
      print(" {0:5d} |".format(ms), end="")
    print()

  def flush_curves(self):
    path_ = os.path.join(self._params['model_prefix'], 'loss_info')
    os.makedirs(path_, exist_ok=True)
    path_ = os.path.join(path_, 'eval_time_range.npy')
    np.save(path_, np.concatenate(self._time_range_eval, axis=0))
    path_ = os.path.join(path_, 'lr_schedule.npy')
    np.save(path_, np.array(self._lr_db_curve))

  def update_learning_rate(self, epoch_step, mode='stepwise'):
    """Update learning rate handler updating only when the mode matches."""
    if self._params['lr_schedule_type'] == mode:
      self._lr_scheduler.step(epoch_step)
      self._writer.add_scalar(
          'learning_rate/lr', self._params['learning_rate'], epoch_step)
      self._lr_db_curve.append([self._params['learning_rate'], epoch_step])
      self._params['learning_rate'] = self._lr_scheduler.get_last_lr()[0]

  @abstractmethod
  def flush_extras(self, epoch, phase):
    pass

  def compute_class_accurracy_sequence(self, class_logits, class_gt):
    # softmax on last dimension and get max on last dimension
    class_pred = torch.argmax(class_logits.softmax(-1), -1)
    accuracy = (class_pred == class_gt).float().sum()
    accuracy = accuracy / class_logits.size()[0]
    return accuracy.item()

  def compute_class_accurracy_instance(self, class_logits, class_gt):
    # softmax on last dimension and get max on last dimension
    tar_seq_len = self._params['target_seq_len']
    class_pred = torch.argmax(class_logits.softmax(-1), -1)
    accuracy = (class_pred == class_gt).float().sum()
    accuracy = accuracy / (class_logits.size()[0]*tar_seq_len)
    return accuracy.item()

  def validation_srnn_ms(self, sample, decoder_pred):
    # the data was flatened from a sequence of size 
    # [n_actions, n_seeds, target_length, pose_size]
    n_actions = len(self._params['action_subset'])
    seq_shape = (n_actions, self._params['eval_num_seeds'], 
        self._params['target_seq_len'], self._params['pose_dim'])
    srnn_gts_euler = sample['decoder_outputs_euler']
    decoder_pred_ = decoder_pred.cpu().numpy()
    decoder_pred_ = decoder_pred_.reshape(seq_shape)
    do_remove = self._params['remove_low_std']
    mean_eval_error_dict = {}

    self.print_table_header()
    eval_ms_mean = []
    for ai, action in enumerate(sample['actions']):
      action = action[0]
      decoder_pred =  decoder_pred_[ai, :, :, :]

      if self._params['dataset'] == 'h36m':
        # seq_len x n_seeds x pose_dim
        decoder_pred = decoder_pred.transpose([1, 0, 2])

      # a list or a vector of length n_seeds
      # each entry of: shape seq_len x complete_pose_dim (H36M == 99)
      srnn_pred_euler = self._eval_dataset_fn.dataset.post_process_to_euler(decoder_pred) 

      # n_seeds x seq_len
      mean_errors = np.zeros((self._params['eval_num_seeds'], 
          self._params['target_seq_len']))

      # Training is done in exponential map or rotation matrix or quaternion
      # but the error is reported in Euler angles, as in previous work [3,4,5] 
      for i in np.arange(self._params['eval_num_seeds']):
        # seq_len x complete_pose_dim (H36M==99)
        eulerchannels_pred = srnn_pred_euler[i]

        # n_seeds x seq_len x complete_pose_dim (H36M==96)
        action_gt = srnn_gts_euler[action]
        # seq_len x complete_pose_dim (H36M==96)
        gt_i = np.copy(action_gt.squeeze()[i].numpy())
        # Only remove global rotation. Global translation was removed before
        gt_i[:, 0:3] = 0

        # here [2,4,5] remove data based on the std of the batch THIS IS WEIRD!
        # (seq_len, 96) - (seq_len, 96)
        idx_to_use = np.where(np.std(gt_i, 0) > 1e-4)[0]
        euc_error = np.power(gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)

        # shape: seq_len
        euc_error = np.sum(euc_error, 1)
        # shape: seq_len
        euc_error = np.sqrt(euc_error)
        mean_errors[i,:] = euc_error

      # This is simply the mean error over the eval_num_seeds examples
      # with shape [eval_num_seeds]
      mean_mean_errors = np.mean(mean_errors, 0)
      mean_eval_error_dict[action] = self.print_range_summary(action, mean_mean_errors)

    return mean_eval_error_dict

  @torch.no_grad()
  def evaluate_h36m(self, current_step, step_time):
    """Evaluation loop."""
    eval_loss = 0

    sample = next(iter(self._eval_dataset_fn))
    # for j, sample in enumerate(self._eval_dataset_fn):
    for k in sample.keys():
      if (k=='decoder_outputs_euler') or (k=='actions'):
        continue
      sample[k] = sample[k].squeeze().to(_DEVICE)

    decoder_pred = self._model(
        sample['encoder_inputs'], sample['decoder_inputs'])

    selection_loss = None
    if self._params['query_selection']:
      prob_mat = decoder_pred[-1][-1]
      selection_loss = self.compute_selection_loss(
          inputs=prob_mat, 
          target=sample['src_tgt_distance']
      )

    pred_class, gt_class = None, None
    if self._params['predict_activity']:
      gt_class = sample['action_ids']  # class per sequence
      pred_class = decoder_pred[1]
      accuracy = self.compute_class_accurracy_sequence(pred_class[-1], gt_class)

    srnn_loss, activity_loss = self.compute_loss(
        inputs=sample['encoder_inputs'],
        target=sample['decoder_outputs'],
        preds=decoder_pred[0],
        class_logits=pred_class,
        class_gt=gt_class
    )

    # [batch_size, sequence_length, pose_dim]
    decoder_pred = decoder_pred[0][-1]
    # [batch_size, sequence_length, pose_dim]
    msre_ = decoder_pred-sample['decoder_outputs']
    # [batch_size, sequence_length]
    msre_ = torch.sqrt(torch.sum(msre_*msre_, dim=-1))
    msre_ = msre_.mean().item()

    eval_loss = (srnn_loss, activity_loss, accuracy) \
        if self._params['predict_activity'] else srnn_loss 
    # run validation on different ranges
    mean_eval_error_dict = self.validation_srnn_ms(sample, decoder_pred)

    self._scalars['ms_eval_loss'] = mean_eval_error_dict
    self._scalars['msre'] = msre_
    self._scalars['eval_selection_loss'] = selection_loss

    return eval_loss

  @torch.no_grad()
  def evaluate_amass(self, current_step, step_time):
    """Evaluation loop."""
    eval_loss = 0.0
    N = len(self._eval_dataset_fn)
    gt_class_ = []
    pred_class_ = []
    total_msre = 0.
    total_srnn = 0.
    total_euc_error = 0.

    for (i, sample) in tqdm.tqdm(enumerate(self._eval_dataset_fn)):
      for k in sample.keys():
        if k=='action_str':
          continue
        if (k=='decoder_outputs_euler') or (k=='actions'):
          continue
        sample[k] = sample[k].to(_DEVICE)

      decoder_pred = self._model(
          sample['encoder_inputs'], 
          sample['decoder_inputs']
      )

      selection_loss = None
      if self._params['query_selection']:
        prob_mat = decoder_pred[-1][-1]
        selection_loss = self.compute_selection_loss(
            inputs=prob_mat, 
            target=sample['src_tgt_distance']
        )

      pred_class, gt_class = None, None
      if self._params['predict_activity']:
        gt_class = sample['action_ids']  # class per sequence
        pred_class = decoder_pred[1]
        accuracy = self.compute_class_accurracy_sequence(pred_class[-1], gt_class)

      srnn_loss, activity_loss = self.compute_loss(
        inputs=sample['encoder_inputs'],
        target=sample['decoder_outputs'],
        preds=decoder_pred[0],
        class_logits=pred_class,
        class_gt=gt_class
      )

      # [batch_size, sequence_length, pose_dim]
      decoder_pred = decoder_pred[0][-1]
      # [batch_size, sequence_length, pose_dim]
      msre_ = decoder_pred-sample['decoder_outputs']
      # [batch_size, sequence_length]
      msre_ = torch.sqrt(torch.sum(msre_*msre_, dim=-1))
      msre_ = msre_.mean().item()
      total_msre += msre_
      eval_loss += srnn_loss

      # run validation on different ranges
     
      # [batch_size, sequence_length, n_joints*3] 
      srnn_gts_euler = sample['decoder_outputs_euler']['unknown'].cpu().numpy()
      # [batch_size, sequence_length, n_joints, 3] 
      srnn_pred_euler = self._eval_dataset_fn.dataset.post_process_to_euler(
          decoder_pred.cpu().numpy())
      B, S, NJ, D = srnn_pred_euler.shape
      srnn_pred_euler = srnn_pred_euler.reshape((B, S, -1))
      # [batch_size, sequence_length, pose_dim]
      euc_error = np.power(srnn_gts_euler - srnn_pred_euler, 2)
      # [batch_size, sequence_length]
      euc_error = np.sum(euc_error, 2)
      # sequence_length
      euc_error = np.mean(np.sqrt(euc_error), 0)
      total_euc_error += euc_error

    selection_loss = selection_loss/N if selection_loss is not None else None
    eval_loss = eval_loss/N
    total_euc_error = total_euc_error / N
    mean_eval_error_dict = self.print_range_summary('unknown', total_euc_error)
    self._scalars['ms_eval_loss'] = {'unknown': mean_eval_error_dict}
    self._scalars['msre'] = total_msre / N
    self._scalars['eval_selection_loss'] = selection_loss

    return eval_loss

  @torch.no_grad()
  def evaluate_nturgbd(self, current_step, dummy_entry=None):
    eval_loss = 0.0
    mAP_all = 0.0
    class_loss = 0.0
    mean_accuracy = 0.0
    N = len(self._eval_dataset_fn)
    gt_class_ = []
    pred_class_ = []

    num_joints = self._params['pose_dim'] // 3
    TP = np.zeros((num_joints,))
    FN = np.zeros((num_joints,))
    MPJPE = np.zeros((num_joints,))

    for (i, sample) in tqdm.tqdm(enumerate(self._eval_dataset_fn)):
      for k in sample.keys():
        if k=='action_str':
          continue
        sample[k] = sample[k].to(_DEVICE)

      decoder_pred = self._model(
          sample['encoder_inputs'], sample['decoder_inputs'])

      pred_class, gt_class = None, None
      if self._params['predict_activity']:
        gt_class = sample['action_ids']  # one label for the sequence
        pred_class = decoder_pred[1]
        decoder_pred = decoder_pred[0]
        gt_class_.append(gt_class.item())
        pred_class_.append(pred_class[-1].cpu().numpy())

      pose_loss, activity_loss = self.compute_loss(
          inputs=sample['encoder_inputs'],
          target=sample['decoder_outputs'],
          preds=decoder_pred,
          class_logits=pred_class,
          class_gt=gt_class
      )

      eval_loss+= pose_loss

    eval_loss /= N
    # compute overall mAP and MPJPE
    mAP_all = TP/(TP + FN)

    print('[INFO] mAP@10cm per joint')
    print(mAP_all)
    mAP_all = np.mean(mAP_all)

    MPJPE = MPJPE/N
    print('[INFO] MPJPE per joint')
    print(MPJPE)
    MPJPE = np.mean(MPJPE)

    self._scalars['mAP'] = mAP_all
    self._scalars['MPJPE'] = MPJPE

    if self._params['predict_activity']:
      class_loss /= N
      pred_class_ = torch.squeeze(torch.from_numpy(np.stack(pred_class_)))
      gt_class_ = torch.from_numpy(np.array(gt_class_))
      print(pred_class_.size(), gt_class_.size())
      accuracy = self.compute_class_accurracy_sequence(pred_class_, gt_class_)
      return (eval_loss, class_loss, accuracy)

    return eval_loss

  def compute_mean_average_precision(self, prediction, target):
    pred = np.squeeze(prediction)
    tgt = np.squeeze(target)
    T, D = pred.shape

    pred = self._eval_dataset_fn.dataset.unormalize_sequence(pred)
    tgt = self._eval_dataset_fn.dataset.unormalize_sequence(tgt)

    # num_frames x num_joints x 3
    pred = pred.reshape((T, -1, 3))
    tgt = tgt.reshape((T, -1, 3))

    # compute the norm for the last axis: (x,y,z) coordinates
    # num_frames x num_joints
    TP = np.linalg.norm(pred-tgt, axis=-1) <= _MAP_TRESH
    TP = TP.astype(int)
    FN = np.logical_not(TP).astype(int)

    # num_joints
    TP = np.sum(TP, axis=0)
    FN = np.sum(FN, axis=0)
    # compute recall for each joint
    recall = TP / (TP+FN)
    # average over joints
    mAP = np.mean(recall)
    return mAP, TP, FN

  
  def compute_MPJPE(self, prediction, target):
    pred = np.squeeze(prediction)
    tgt = np.squeeze(target)
    T, D = pred.shape

    pred = self._eval_dataset_fn.dataset.unormalize_sequence(pred)
    tgt = self._eval_dataset_fn.dataset.unormalize_sequence(tgt)

    # num_frames x num_joints x 3
    pred = pred.reshape((T, -1, 3))
    tgt = tgt.reshape((T, -1, 3))

    # compute the norm for the last axis: (x,y,z) coordinates
    # num_frames x num_joints
    norm = np.linalg.norm(pred-tgt, axis=-1)

    # num_joints
    MPJPE = np.mean(norm, axis=0)
    return MPJPE


def dataset_factory(params):
  """Defines the datasets that will be used for training and validation."""
  train_dataset = h36mdataset_fn.H36MDataset(params, mode='train')
  train_dataset_fn = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=params['batch_size'],
      shuffle=True,
      num_workers=4,
      collate_fn=h36mdataset_fn.collate_fn,
      drop_last=True
  )

  eval_dataset = h36mdataset_fn.H36MDataset(
      params, 
      mode='eval',
      norm_stats=train_dataset._norm_stats
  )
  eval_dataset_fn = torch.utils.data.DataLoader(
      eval_dataset,
      batch_size=1,
      shuffle=True,
      num_workers=1,
      drop_last=True
  ) 

  return train_dataset_fn, eval_dataset_fn, train_dataset.get_pose_dim()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default=None)
  parser.add_argument('--action', type=str, default=None)
  parser.add_argument('--use_one_hot',  action='store_true')
  parser.add_argument('--source_seq_len', type=int, default=50)
  parser.add_argument('--target_seq_len', type=int, default=25)
  # parser.add_argument('--input_size', type=int, default=55)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--max_epochs', type=int, default=3000)
  parser.add_argument('--steps_per_epoch', type=int, default=200)
  parser.add_argument('--learning_rate', type=float, default=0.005)
  parser.add_argument('--optimizer_fn', type=str, default='adam')
  parser.add_argument('--warmup_epochs', type=int, default=30)
  parser.add_argument('--remove_low_std', action='store_true')
  args = parser.parse_args()
  
  params = vars(args)
  print(params)
  params['action_subset'] = [args.action]
  params['virtual_dataset_size'] = args.steps_per_epoch*args.batch_size

  train_dataset_fn, eval_dataset_fn = dataset_factory(params)

  model_fn = Seq2SeqModelFn(
    params, 
    train_dataset_fn=train_dataset_fn, 
    eval_dataset_fn=eval_dataset_fn
  )

  model_fn.train()



