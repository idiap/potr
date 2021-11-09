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

"""Human 3.6 M dataset management.

The data is preprocessed and sampled just as in [3]. The code here has been
adapted to be an isolated module in Pytorch [4].

[1] See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
[2] https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
[3] https://arxiv.org/abs/1705.02445
[4] https://github.com/una-dinosauria/human-motion-prediction
[5] https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
[6] https://github.com/eth-ait/spl/blob/master/preprocessing/preprocess_h36m.py
"""


import numpy as np
import os
import json
import argparse
import logging
import sys
import copy

import torch
import torch.nn.functional as F

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import utils.utils as utils
import visualize.viz as viz

_ALL_ACTIONS = [
    "directions", "discussion", "eating", "greeting", "phoning",
    "posing", "purchases", "sitting", "sittingdown", "smoking",
    "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"
]

# SOA consider only 21 major joints
_MAJOR_JOINTS = [
    0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 24, 25, 26, 27
]
_NMAJOR_JOINTS = len(_MAJOR_JOINTS)
_NH36M_JOINTS = 32

_MOCAP_DIM = 99
_MIN_STD = 1e-4
_DIM_AA = 3
_DIM_RT = 9


def copy_uniform_scan(input_sequence, query_sequence):
  """Copies elements from input_sequence to query_sequence.

  Args:                                                                          
    input_sequence: [:, source_length, pose_dim]
    query_sequence: [:, target_length, pose_dim]
  """
  # query_sequence already has padded the first query pose, just padd the other
  if len(input_sequence.shape) == 3:
    _, S, D = input_sequence.shape
    _, T, _ = query_sequence.shape
  else:
    S, D = input_sequence.shape
    T, _ = query_sequence.shape

  t = T-2
  i = S-1
  while t>=0:
    if len(input_sequence.shape) == 3:
      query_sequence[:, t] = copy.deepcopy(input_sequence[:, i])
    else:
      query_sequence[t] = copy.deepcopy(input_sequence[i])
    t-=1
    i-=1


def assert_valid_inputs(inputs, valid_entries):
  for inp in inputs:
    assert inp in valid_entries, '{} is not a valid action.'.format(inp)


def _find_indices_srnn(data, action):
  """Find the same action indices as in SRNN. See [1]."""
  # Used a fixed dummy seed, following [5]
  SEED = 1234567890
  rng = np.random.RandomState(SEED)

  subject = 5
  subaction1 = 1
  subaction2 = 2

  T1 = data[(subject, action, subaction1)].shape[0]
  T2 = data[(subject, action, subaction2)].shape[0]
  prefix, suffix = 50, 100

  idx = []
  idx.append(rng.randint(16,T1-prefix-suffix))
  idx.append(rng.randint(16,T2-prefix-suffix))
  idx.append(rng.randint(16,T1-prefix-suffix))
  idx.append(rng.randint(16,T2-prefix-suffix))
  idx.append(rng.randint(16,T1-prefix-suffix))
  idx.append(rng.randint(16,T2-prefix-suffix))
  idx.append(rng.randint(16,T1-prefix-suffix))
  idx.append(rng.randint(16,T2-prefix-suffix))
  return idx 


def collate_fn(batch):
  """Collate function for data loaders."""
  e_inp = torch.from_numpy(np.stack([e['encoder_inputs'] for e in batch]))
  d_inp = torch.from_numpy(np.stack([e['decoder_inputs'] for e in batch]))
  d_out = torch.from_numpy(np.stack([e['decoder_outputs'] for e in batch]))
  action_id = torch.from_numpy(np.stack([e['action_id'] for e in batch]))
  a_instance = torch.from_numpy(np.stack([e['action_id_instance'] for e in batch]))
  action = [e['actions'] for e in batch]
  d = torch.from_numpy(np.stack([e['src_tgt_distance'] for e in batch]))

  batch_ = {
      'encoder_inputs': e_inp,
      'decoder_inputs': d_inp,
      'decoder_outputs': d_out,
      'actions': action,
      'action_ids': action_id,
      'action_id_instance': a_instance,
      'src_tgt_distance': d
  }

  return batch_


class H36MDataset(torch.utils.data.Dataset):
  """Implements Human3.6M action dataset for motion prediction."""

  def __init__(self, 
              params=None,
              mode='train', 
              norm_stats=None,
              **kwargs):
    """Dataset factory initialization.

    Args:
      params: A dictionary with keys: `data_path` a string pointing the path
        to the dataset location, `action_subset` a list of strings listing 
        required actions to train and test.
      mode: Phase of training stage. Should be `train` or `eval`.
      norm_stats: Normalization parameters. These should be computing only
        for training and provided only when `mode=eval`.
    """
    super(H36MDataset, self).__init__(**kwargs)
    self._params = params
    self._train_ids = [1,6,7,8,9,11] 
    self._test_ids = [5]
    self._test_subject = 5
    self._test_n_seeds = params['eval_num_seeds']
    self._action_defs = []
    self._norm_stats = {}
    self._data = {}
    self._mode = mode
    self._monitor_action = 'walking'

    # select what subjects id to use depending on operation mode
    self._data_ids = self._train_ids if mode == 'train' else self._test_ids

    for k, v in self._params.items():
      print('[INFO] (H36MDataset)  {}: {}'.format(k, v))
    print('[INFO] (H36MDataset)  mode: {}'.format(self._mode))
    print('[INFO] (H36MDataset)  data_ids: {}'.format(self._data_ids))

    if mode == 'eval':
      assert norm_stats is not None, \
          'Normalization stats should be provided for evaluation dataset.'
      self._norm_stats = norm_stats

    self._action_ids = {
        self._params['action_subset'][i]:i 
        for i in range(len(self._params['action_subset']))
    }

    self._all_data = self.load_data()
    self._data_keys = list(self._data.keys())
    print('[INFO] (H36MDataset) ', self._data_keys)

  def __call__(self, subject_id, action):
    """Create dataset for the given subject id."""
    return H36MDataset(self._params)

  def __len__(self):
    return max(len(self._data_keys), self._params['virtual_dataset_size'])

  def convert_to_format(self, poses):
    """Convert poses to required format."""
    pose_format = self._params['pose_format']
    if pose_format == 'expmap':
      return poses
    elif pose_format == 'rotmat':
      return utils.expmap_to_rotmat(poses)
    else:
      raise ValueError('Format {} unknown!'.format(pose_format))

  def preprocess_sequence(self, action_sequence):
    """Seleection the good joints and convert to required format.
    Args:
      action_sequence: [n_frames, 96]
    """
    total_frames, D = action_sequence.shape
    # total_framesx32x3
    data_sel = action_sequence.reshape((total_frames, -1, 3))
    # total_framesx21x3
    data_sel = data_sel[:, _MAJOR_JOINTS]
    # total_framesx21xformat_dim
    data_sel = self.convert_to_format(data_sel)
    # total_frames x n_joints*dim_per_joint
    data_sel = data_sel.reshape((total_frames, -1))

    return data_sel

  def compute_difference_matrix(self, src_seq, tgt_seq):
    """Computes a matrix of euclidean difference between sequences.

    Args:
      src_seq: Numpy array of shape [src_len, dim].
      tgt_seq: Numpy array of shape [tgt_len, dim].

    Returns:
      A matrix of shape [src_len, tgt_len] with euclidean distances.
    """
    src_len = src_seq.shape[0] # M
    tgt_len = tgt_seq.shape[0] # N

    distance = np.zeros((src_len, tgt_len), dtype=np.float32)
    for i in range(src_len):
      for j in range(tgt_len):
        distance[i, j] = np.linalg.norm(src_seq[i]-tgt_seq[j])

    row_sums = distance.sum(axis=1)
    distance_norm = distance / row_sums[:, np.newaxis]
    distance_norm = 1.0 - distance_norm

    return distance, distance_norm
    

  def _get_item_train(self):
    """Get item for the training mode."""
    idx = np.random.choice(len(self._data_keys), 1)[0]

    the_key = self._data_keys[idx]
    action = the_key[1]
    source_seq_len = self._params['source_seq_len']
    target_seq_len = self._params['target_seq_len']
    input_size = self._data_dim
    pose_size = self._pose_dim
    total_frames = source_seq_len + target_seq_len

    src_seq_len = source_seq_len - 1
    if self._params['include_last_obs']:
      src_seq_len += 1

    encoder_inputs = np.zeros((src_seq_len, input_size), dtype=np.float32)
    decoder_inputs = np.zeros((target_seq_len, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((target_seq_len, pose_size), dtype=np.float32)

    N, _ = self._data[the_key].shape
    start_frame = np.random.randint(16, N-total_frames)
    # total_framesxn_joints*joint_dim
    data_sel = self._data[the_key][start_frame:(start_frame+total_frames), :]

    encoder_inputs[:, 0:input_size] = data_sel[0:src_seq_len,:]
    decoder_inputs[:, 0:input_size] = \
        data_sel[src_seq_len:src_seq_len+target_seq_len, :]
    # source_seq_len = src_seq_len + 1
    decoder_outputs[:, 0:pose_size] = data_sel[source_seq_len:, 0:pose_size]

    if self._params['pad_decoder_inputs']:
      query = decoder_inputs[0:1, :]
      decoder_inputs = np.repeat(query, target_seq_len, axis=0)
      #if self._params['copy_method'] == 'uniform_scan':
      #  copy_uniform_scan(encoder_inputs, decoder_inputs)

    distance, distance_norm = self.compute_difference_matrix(
        encoder_inputs, decoder_outputs
    )

    return {
        'encoder_inputs': encoder_inputs, 
        'decoder_inputs': decoder_inputs, 
        'decoder_outputs': decoder_outputs,
        'actions': action,
        'action_id': self._action_ids[action],
        'action_id_instance': [self._action_ids[action]]*target_seq_len,
        'src_tgt_distance': distance
    }
    
  def _get_item_eval(self):
    """Sample a batch for evaluation along with euler angles."""
    src_len = self._params['source_seq_len'] - 1
    tgt_len = self._params['target_seq_len']
    pose_size = self._pose_dim # self._params['input_size']
    size = self._data_dim
    if self._params['include_last_obs']:
      src_len += 1

    batch = [self._sample_batch_eval(action) 
        for action in self._params['action_subset']]
    euler_data = {e['actions']: e['decoder_outputs_euler'] for e in batch}
    batch = collate_fn(batch)
    decoder_inputs = batch['decoder_inputs'].view(-1, tgt_len, size)
    decoder_outputs = batch['decoder_outputs'].view(-1, tgt_len, size)
    encoder_inputs = batch['encoder_inputs'].view(-1, src_len, size)
    distance = batch['src_tgt_distance'].view(-1, src_len, tgt_len)

    action_ids = np.array([
        np.repeat(self._action_ids[a], self._test_n_seeds).tolist()
        for a in self._params['action_subset']
    ]).ravel()
    action_ids = torch.from_numpy(action_ids)

    return {
        'encoder_inputs': encoder_inputs,
        'decoder_inputs': decoder_inputs,
        'decoder_outputs': decoder_outputs,
        'decoder_outputs_euler': euler_data,
        'actions': self._params['action_subset'],
        'action_ids': action_ids,
        'action_id_instance': batch['action_id_instance'],
        'src_tgt_distance': distance
    }

  def _sample_batch_eval(self, action):
    """Get a random batch of data from the specified bucket, prepare for step.

    The test is always done with subject 5 and takes 8 random seeds. The test 
    data is a dictionary with k:v, k=((subject, action, subsequence, 'even')),
    v=nxd matrix with a sequence of poses. The function samples the test seeds
    for reproducing SRNN's sequence subsequence selection as done in [2]

    Args:
      action: the action to load data from
    """
    src_seq_len= self._params['source_seq_len']
    tgt_seq_len = self._params['target_seq_len']
    n_seeds = self._test_n_seeds
    input_size = self._data_dim
    pose_size = self._pose_dim
    src_seq_len = src_seq_len - 1
    if self._params['include_last_obs']:
      src_seq_len += 1
    # Compute the number of frames needed
    source_seq_len_complete = self._params['source_seq_len']
    if source_seq_len_complete < 50:
      source_seq_len_complete = 50

    if action not in _ALL_ACTIONS:
      raise ValueError("Unrecognized action {0}".format(action))

    frames = _find_indices_srnn(self._data, action)
    seeds = [(action, (i%2)+1, frames[i]) for i in range(self._test_n_seeds)]

    encoder_inputs = np.zeros((n_seeds, src_seq_len, input_size), dtype=np.float32)
    decoder_inputs = np.zeros((n_seeds, tgt_seq_len, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((n_seeds, tgt_seq_len, pose_size), dtype=np.float32)
    decoder_outputs_euler = np.zeros((n_seeds, tgt_seq_len, 96), dtype=np.float32)
    action_id_instance= np.zeros((n_seeds, tgt_seq_len), dtype=np.int64)
    distance = np.zeros((n_seeds, src_seq_len, tgt_seq_len), dtype=np.float32)

    for i in range(self._test_n_seeds):
      _, subsequence, idx = seeds[i]
      idx = idx + source_seq_len_complete
      the_key = (self._test_subject, action, subsequence)
      data_sel = self._data[the_key]
      data_sel = data_sel[(idx-src_seq_len):(idx+tgt_seq_len) , :]
      data_sel_srnn = self._data_srnn[the_key]
      data_sel_srnn = data_sel_srnn[(idx-src_seq_len):(idx+tgt_seq_len) , :]

      encoder_inputs[i, :, :] = data_sel[0:src_seq_len, :]
      decoder_inputs[i, :, :] = data_sel[src_seq_len:(src_seq_len+tgt_seq_len), :]
      decoder_outputs[i, :, :] = data_sel[src_seq_len:, 0:pose_size]
      action_id_instance[i, :] = self._action_ids[action]
      distance[i] = self.compute_difference_matrix(
          encoder_inputs[i], decoder_outputs[i])[0]
      # tgt_seq_len x 96
      decoder_outputs_srnn = np.expand_dims(data_sel_srnn[src_seq_len:], axis=0)
      # tgt_seq_len x 32 x 3
      euler = decoder_outputs_srnn.reshape((tgt_seq_len, -1, 3))
      euler = utils.expmap_to_euler(euler)
      # tgt_seq_len x 96
      decoder_outputs_euler[i] = euler.reshape((tgt_seq_len, -1))

      if self._params['pad_decoder_inputs']:
        query = decoder_inputs[i, 0:1, :]
        decoder_inputs[i, :, :] = np.repeat(query, tgt_seq_len, axis=0)
        #if self._params['copy_method'] == 'uniform_scan':
        #  copy_uniform_scan(encoder_inputs, decoder_inputs)

      if self._params['pad_decoder_inputs_mean']:
        query_mean = np.mean(encoder_inputs[i], axis=0)[np.newaxis,...]
        decoder_inputs[i, :, :] = np.repeat(query_mean, tgt_seq_len, axis=0)

    return {
        'encoder_inputs': encoder_inputs, 
        'decoder_inputs': decoder_inputs, 
        'decoder_outputs': decoder_outputs,
        'decoder_outputs_euler': decoder_outputs_euler,
        'actions': action,
        'action_id': self._action_ids[action],
        'action_id_instance': action_id_instance,
        'src_tgt_distance': distance
    }

  def __getitem__(self, idx):
    """Get item in dataset according to random sampling. `idx` is ignored."""
    if self._mode == 'train':
      return self._get_item_train()
    return self._get_item_eval()

  def get_pose_dim(self):
    """Returns the pose dimension as a flattened vector."""
    return self._pose_dim

  def load_action_defs(self):
    """Loads required actions to be trained with."""
    json_config = os.path.join(self._params['data_path'], 'action_defs.json')
    self._action_defs_gt = json.load(open(json_config))
    if self._params['action_subset'] is not None:
      assert_valid_inputs(self._params['action_subset'], self._action_defs_gt)
      self._action_defs = [a for a in self._params['action_subset'] 
                           if a in self._action_defs_gt]

    logging.info("Loading actions: {}".format(self._action_defs))
    
  def load_data(self):
    """Loads all the H3.6M dataset into memory."""
    self._data = {} # {id_:{} for id_ in self._train_ids}
    self._data_srnn = {} # {id_:{} for id_ in self._train_ids}

    self.load_action_defs()
    self._n_actions = len(self._action_defs)
    file_prefix = "{}/S{}/{}_{}.npy"
    dataset_path = os.path.join(self._params['data_path'], 'dataset')

    all_dataset = []
    all_dataset_ = []
    for s_id in self._data_ids:
      for a_idx, action in enumerate(self._action_defs):
        for sact in [1, 2]:
          file_ = file_prefix.format(dataset_path, s_id, action, sact)
          action_sequence = np.load(file_)
          n_frames, dims = action_sequence.shape
          even_idx = range(0, n_frames, 2)
          # get the even sequence in shape [len(even_idx), dims]
          action_sequence = action_sequence[even_idx, :]
          entry_key = (s_id, action, sact)
          all_dataset_.append(action_sequence)
          # Remove the first three dimensions (root position) as in [6]
          # n_framesx96
          action_sequence_ = action_sequence[:, 3:]
          # preprocess by keeping only the major joints and seleect the format
          # n_framesx96 -> n_framesxnjoints*joint_dim
          action_sequence = self.preprocess_sequence(action_sequence_)
          self._data[entry_key] = action_sequence
          self._data_srnn[entry_key] = action_sequence_
          all_dataset.append(action_sequence)

    all_dataset = np.concatenate(all_dataset, axis=0)
    self._all_dataset = np.concatenate(all_dataset_, axis=0)
    print('[INFO] ({}) All dataset shape: {}'.format(self.__class__.__name__, all_dataset.shape))
    # compute normalization statistics only when in training phase
    if self._mode == 'train':
      self.compute_norm_stats(all_dataset)
    self.normalize_data()

    self._pose_dim = self._norm_stats['std'].shape[-1]
    self._data_dim = self._pose_dim

    thisname = self.__class__.__name__
    print('[INFO] ({}) Pose dim: {} Data dim: {}'.format(
        thisname, self._pose_dim, self._data_dim))

    return all_dataset

  def normalize_data(self):
    """Data normalization with mean and std removing dimensions with low std."""
    for k in self._data.keys():
      tmp_data = self._data[k]
      tmp_data_ = self._data[k]
      tmp_data = tmp_data - self._norm_stats['mean']
      tmp_data = np.divide(tmp_data, self._norm_stats['std'])
      self._data[k] = tmp_data

  def compute_norm_stats(self, data):
    """Compute normalization statistics and slice according to MAJOR_JOINTS."""
    self._norm_stats = {}
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[np.where(std<_MIN_STD)] = 1

    self._norm_stats['mean'] = mean.ravel()
    self._norm_stats['std'] = std.ravel()
    # stats are already comptued for major joints
    # self._norm_stats['std_used'] = std[_MAJOR_JOINTS]
    # self._norm_stats['mean_used'] = mean[_MAJOR_JOINTS]

  def convert_to_euler(self, action_sequence_, org_format='expmap', is_normalized=True):
    """Convert the input exponential maps to euler angles.

    Args:
      action_sequence: Pose exponential maps [batch_size, sequence_length, pose_size].
        The input should not contain the one hot encoding in the vector.
    """
    B, S, D = action_sequence_.shape
    # first unnormalize data to then convert to euler
    if is_normalized:
      action_sequence_ = action_sequence_*self._norm_stats['std'] + self._norm_stats['mean']
    rotmats = action_sequence_.reshape((B*S, _NMAJOR_JOINTS, -1))
    if org_format == 'expmap':
      rotmats = utils.expmap_to_rotmat(rotmats)

    euler_maps = utils.rotmat_to_euler(rotmats)
    euler_maps = euler_maps.reshape((B, S, -1))

    return euler_maps

  def to_euler(self, pred_sequence):
    """Transform predicted sequence to euler angles.
    Args:
      pred_sequence: [sequence_length, dim].
    """
    S, D = pred_sequence.shape
    pred_sequence = pred_sequence.reshape((1, S, D))
    # 1 x seq_len, 3*n_joints
    pred_sequence = self.convert_to_euler(pred_sequence, self._params['pose_format'])
    return np.squeeze(pred_sequence)

  def post_process_to_euler(self, norm_seq):
    """Converts to euler angles and pad with zeros the minor joints.
    Args:
      norm_seq: A numpy array. Normalized sequence [batch_size, seq_length, 
        n_major_joints*dof]
    """
    batch_size, seq_length, D = norm_seq.shape
    # batch_size x seq_length x n_major_joints*dof
    euler_seq = self.convert_to_euler(norm_seq, self._params['pose_format'])
    # batch_size x seq_length x n_major_joints x dof (or joint dim)
    euler_seq = euler_seq.reshape((batch_size, seq_length, _NMAJOR_JOINTS, 3))
    p_euler_padded = np.zeros([batch_size, seq_length, _NH36M_JOINTS, 3])
    p_euler_padded[:, :, _MAJOR_JOINTS] = euler_seq
    # batch_size x seq_length x _NH36M_JOINTS*3
    p_euler_padded = np.reshape(p_euler_padded, [batch_size, seq_length, -1])
    return p_euler_padded

  def unnormalize_pad_data_to_expmap(self, norm_seq):
    """Unnormalize data and pads with zeros the minor joints.

    Args:
      norm_seq: A numpy array. Normalized sequence [batch_size, seq_length, 
        n_major_joints*dof]

    Returnrs:
      Numpy array of shape [batch_size, seq_length, 99]
    """
    batch_size, seq_length, D = norm_seq.shape
    dof = D//_NMAJOR_JOINTS

    # unnormalize input sequence
    sequence = norm_seq*self._norm_stats['std'] + self._norm_stats['mean']

    # convert to expmaps in case pose format is rotation matrices
    if self._params['pose_format'] == 'rotmat':
      sequence = sequence.reshape((batch_size*seq_length, _NMAJOR_JOINTS, 9))
      sequence = utils.rotmat_to_expmap(sequence)
      # batch_size x seq_length x 63
      sequence = sequence.reshape((batch_size, seq_length, -1))
      dof = 3

    # batch_size x seq_length x n_major_joints x dof (or joint dim)
    sequence = sequence.reshape((batch_size, seq_length, _NMAJOR_JOINTS, dof))
    # pad data comprising all the joints and the root position
    # batch_size x seq_length x 33 x dof (or joint dim)
    seq_padded = np.zeros([batch_size, seq_length, _NH36M_JOINTS+1, dof])
    seq_padded[:, :, np.array(_MAJOR_JOINTS)+1] = sequence 
    # batch_size x seq_length x 99
    seq_padded = np.reshape(seq_padded, [batch_size, seq_length, -1])
    return seq_padded


def visualize_sequence(action_sequence, data_path, prefix=None, colors=None):
  """Visualize action sequence.

  Args:
    action_sequence: Numpy array of shape [1, seq_len, l]
    dataset: H36MDataset object.
  """
  # [1, 49, 63]

  if colors is None:
    colors=["#3498db", "#e74c3c"]

  sequence = action_sequence
  parent, offset, rot_ind, exp_map_ind = utils.load_constants(data_path)

  # [seq_len, pose_dim]
  sequence = sequence[0]
  ## [seq_len, 1, pose_dim]
  # nframes should be the same size as target_length
  nframes = sequence.shape[0]

  expmap = utils.revert_coordinate_space(sequence, np.eye(3), np.zeros(3))
  # create data without the root joint 
  xyz_data = np.zeros((nframes, 96))

  for i in range(nframes):
    pose = expmap[i, :]
    xyz_data[i, :] = utils.compute_forward_kinematics(
        pose, parent, offset, rot_ind, exp_map_ind)

  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = plt.gca(projection='3d')
  ob =  viz.Ax3DPose(ax, lcolor=colors[0], rcolor=colors[1])

  for i in range(nframes):
    ob.update(xyz_data[i, :], lcolor=colors[0], rcolor=colors[1])
    plt.show(block=False)
    if prefix!= None:
      plt.savefig(prefix+'_%08d.png'%i, transparent=True)
    fig.canvas.draw()


def dataset_factory(params):
  """Defines the datasets that will be used for training and validation."""
  if 'all' in params['action']:
    params['action_subset'] = _ALL_ACTIONS
  else:
    params['action_subset'] = params['action']

  params['num_activities'] = len(params['action_subset'])
  params['virtual_dataset_size'] = params['steps_per_epoch']*params['batch_size']
  params['n_joints'] = _NMAJOR_JOINTS

  train_dataset = H36MDataset(params, mode='train')
  train_dataset_fn = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=params['batch_size'],
      shuffle=True,
      num_workers=4,
      collate_fn=collate_fn,
      drop_last=True
  )

  eval_dataset = H36MDataset(
      params, 
      mode='eval',
      norm_stats=train_dataset._norm_stats
  )
  eval_dataset_fn = torch.utils.data.DataLoader(
      eval_dataset,
      batch_size=1,
      shuffle=True,
      num_workers=1,
      drop_last=True,
#      collate_fn=collate_fn,
  ) 

  return train_dataset_fn, eval_dataset_fn


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default=None)
  parser.add_argument('--action', type=str, nargs='*', default=None)
  parser.add_argument('--include_last_obs',  action='store_true')
  parser.add_argument('--pad_decoder_inputs',  action='store_true')
  parser.add_argument('--pad_decoder_inputs_mean',  action='store_true')
  parser.add_argument('--source_seq_len', type=int, default=50)
  parser.add_argument('--target_seq_len', type=int, default=25)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--pose_format', type=str, default='expmap')
  parser.add_argument('--steps_per_epoch', type=int, default=200)
  parser.add_argument('--eval_num_seeds', type=int, default=8)
  args = parser.parse_args()

  params = vars(args)
  dataset_t, dataset_v = dataset_factory(params)

#  sample = next(iter(dataset_t))
#  sequence = np.squeeze(sample['decoder_outputs'].cpu().numpy())
#  sequence = dataset_v.dataset.unnormalize_pad_data_to_expmap(sequence)
#  visualize_sequence(sequence[0:1], args.data_path)

  import matplotlib.pyplot as plt
  import matplotlib
  from scipy.optimize import linear_sum_assignment

  for i, sample in enumerate(dataset_t):
    print(sample['src_tgt_distance'].size())
    # softmax by column
    d = F.softmax(-sample['src_tgt_distance'][0], dim=1)
    # softmax by rows
    d2 = F.softmax(-sample['src_tgt_distance'][0], dim=0)
    indices = torch.argmax(d, 0).cpu().numpy()
    #indices = linear_sum_assignment(d, maximize=True)
    print(indices, len(indices))
    
    # print(np.argmin(sample['src_tgt_distance'][0].cpu().numpy(), 1))
    
    fig, ax = plt.subplots(figsize=(20,10))
    ax.matshow(d)
    plt.ylabel("")
    plt.xlabel("")
    fig.tight_layout()
    #plt.show()
    name = 'distances/%s_.png'%(i)
    plt.savefig(name)
    plt.close()


    fig, ax = plt.subplots(figsize=(20,10))
    ax.matshow(d2)
    plt.ylabel("")
    plt.xlabel("")
    fig.tight_layout()
    #plt.show()
    name = 'distances/rows_%s_.png'%(i)
    plt.savefig(name)
    plt.close()

