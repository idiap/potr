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

"""Pytorch dataset of skeletons for the NTU-RGB+D [1] dataset.


[1] http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
[2] https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf
"""


import os
import sys
import numpy as np
import torch
import argparse
import tqdm

# tran subjects id can be found in [2]
_TRAIN_SUBJECTS = [
    1, 2, 4, 5, 8, 9, 13, 14, 15,16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
_TEST_SUBJECTS = [x for x in range(1, 40) if x not in _TRAIN_SUBJECTS]

# the joints according to [2] in 1-base
# 1-base of the spine 2-middle of the spine 3-neck 4-head 5-left shoulder 
# 6-left elbow 7-left wrist 8-left hand 9-right shoulder 10-right elbow
# 11-right wrist 12-right hand 13-left hip 14-left knee 15-left ankle 
# 16-left foot 17-right hip 18-right knee 19-right ankle 20-right foot 
# 21-spine 22-tip of the left hand 23-left thumb 24-tip of the right 
# hand 25-right thumb
# here set the joint indices in base 0
_MAJOR_JOINTS = [x-1 for x in 
    [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21]
]
_NMAJOR_JOINTS = len(_MAJOR_JOINTS)
_SPINE_ROOT = 0
_MIN_STD = 1e-4
# NTURGB+D contains 60 actions
_TOTAL_ACTIONS = 60
_MIN_REQ_FRAMES = 65


def collate_fn(batch):
  """Collate function for data loaders."""
  e_inp = torch.from_numpy(np.stack([e['encoder_inputs'] for e in batch]))
  d_inp = torch.from_numpy(np.stack([e['decoder_inputs'] for e in batch]))
  d_out = torch.from_numpy(np.stack([e['decoder_outputs'] for e in batch]))
  action_id = torch.from_numpy(np.stack([e['action_id'] for e in batch]))
  action = [e['action_str'] for e in batch]

  batch_ = {
      'encoder_inputs': e_inp,
      'decoder_inputs': d_inp,
      'decoder_outputs': d_out,
      'action_str': action,
      'action_ids': action_id
  }

  return batch_



def load_action_labels(data_path):
  data_labels = []
  with open(os.path.join(data_path, 'action_labels.txt')) as file_:
    for line in file_:
      data_labels.append(line.strip())

  return data_labels


def get_activity_from_file(path_file):
  # The pattern is SsssCcccPpppRrrrAaaa.skeleton
  pattern = path_file.split('/')[-1].split('.')[0]
  setup_id= int(pattern[1:4])
  camera_id = int(pattern[5:8])
  subject_id = int(pattern[9:12])
  replication_id = int(pattern[13:16])
  activity_id = int(pattern[17:])

  return (setup_id, camera_id, subject_id, replication_id, activity_id)


def select_fold_files(path_to_data, skip_files_path):
  all_files = [x for x in os.listdir(path_to_data) if x.endswith('skeleton')]

  with open(skip_files_path) as file_:
    skip_files = [line.strip() for line in file_]

  training_files = []
  test_files = []

  for path_file in all_files:
    if path_file.split('.')[0] in skip_files:
      print('Skiping file:', path_file)
      continue

    seq_info = get_activity_from_file(path_file)
    if seq_info[2] in _TRAIN_SUBJECTS:
      training_files.append(path_file)
    else:
      test_files.append(path_file)

  return training_files, test_files


def save_fold_files(path_to_data, output_path, skip_files_path):
  training_files, test_files = select_fold_files(path_to_data, skip_files_path)

  val_idx = np.random.choice(
      len(training_files), int(len(training_files)*0.05), replace=False)

  training_files = [training_files[i] 
      for i in range(len(training_files)) if i not in val_idx]

  val_files = [training_files[i]
      for i in range(len(training_files)) if i in val_idx]

  with open(os.path.join(output_path, 'training_files.txt'), 'w') as file_:
    for f in training_files:
      print(f, file=file_)

  with open(os.path.join(output_path, 'testing_files.txt'), 'w') as file_:
    for f in test_files:
      print(f, file=file_)

  with open(os.path.join(output_path, 'validation_files.txt'), 'w') as file_:
    for f in val_files:
      print(f, file=file_)
  

def read_sequence_kinect_skeletons(path_file):
  """Reads the text file provided in the 
  """
  fid = open(path_file, 'r')

  seq_info = get_activity_from_file(path_file)
  # first line is the number of frames
  framecount = int(fid.readline().strip())
  bodies = {}

  for i in range(framecount):
    bodycount = int(fid.readline().strip())
    for b in range(bodycount):
      # traccking ID of the skeleton
      line = fid.readline().strip().split(' ')
      body_id = int(line[0])
      arrayint = [int(x) for x in line[1:7]]
      lean = [float(x) for x in line[7:9]]
      tracking_state = int(line[-1])

      #number of joints
      joint_count = int(fid.readline().strip())
      joints = []

      for j in range(joint_count):
        line = fid.readline().strip().split(' ')
        # 3D location of the joint
        joint_3d = [float(x) for x in line[0:3]]

        # 2D depth location of joints
        joint_2d_depth = [float(x) for x in line[3:5]]
        # 2D color location of joints
        joint_2d_color = [float(x) for x in line[5:7]]
        # orientation of joints (?)
        joint_orientation = [float(x) for x in line[7:11]]
        # tracking state
        joint_track_state = int(line[-1])

        joints.append(joint_3d)

      if body_id in list(bodies.keys()):
        bodies[body_id].append(np.array(joints, dtype=np.float32))
      else:
        bodies[body_id] = [np.array(joints, dtype=np.float32)]

  for k, v in bodies.items():
    bodies[k] = np.stack(v)

  return bodies, seq_info 


def select_sequence_based_var(action_sequence_dict):
  """Selects the actor in sequence based on the sum of variance of X, Y, Z."""
  larger_var = -1
  selected_key = None
  for k, v in action_sequence_dict.items():
    var = np.var(v, axis=-1)
    sum_var = np.sum(var)
    if sum_var > larger_var:
      larger_var = sum_var
      selected_key = k

  return action_sequence_dict[selected_key]


class NTURGDDatasetSkeleton(torch.utils.data.Dataset):
  def __init__(self, params=None, mode='train'):
    super(NTURGDDatasetSkeleton, self).__init__()
    self._params = params
    self._mode = mode
    thisname = self.__class__.__name__
    self._monitor_action = 'walking'
    for k, v in params.items():
      print('[INFO] ({}) {}: {}'.format(thisname, k, v))

    data_path = self._params['data_path']
    self._action_str = load_action_labels(data_path)

    self._fold_file = ''
    if self._mode.lower() == 'train':
      self._fold_file = os.path.join(data_path, 'training_files.txt')
    elif self._mode.lower() == 'eval':
      self._fold_file = os.path.join(data_path, 'validation_files.txt')
    elif self._mode.lower() == 'test':
      self._fold_file = os.path.join(data_path, 'testing_files.txt')
    else:
      raise ValueError('Unknown launching mode: {}'.format(self._mode))

    self.load_data()

  def read_fold_file(self, fold_file):
    files = []
    with open(fold_file) as file_:
      for line in file_:
        files.append(line.strip())

    return files

  def compute_norm_stats(self, data):
    self._norm_stats = {}
    mean = np.mean(data, axis=0)
    std = np.mean(data, axis=0)
    std[np.where(std<_MIN_STD)] = 1

    self._norm_stats['mean'] = mean.ravel()
    self._norm_stats['std'] = std.ravel()

  def load_compute_norm_stats(self, data):
    mean_path = os.path.join(self._params['data_path'], 'mean.npy')
    std_path = os.path.join(self._params['data_path'], 'std.npy')
    thisname = self.__class__.__name__
    self._norm_stats = {}

    if os.path.exists(mean_path):
      print('[INFO] ({}) Loading normalization stats!'.format(thisname))
      self._norm_stats['mean'] = np.load(mean_path)
      self._norm_stats['std'] = np.load(std_path)
    elif self._mode == 'train':
      print('[INFO] ({}) Computing normalization stats!'.format(thisname))
      self.compute_norm_stats(data)
      np.save(mean_path, self._norm_stats['mean'])
      np.save(std_path, self._norm_stats['std'])
    else:
      raise ValueError('Cant compute statistics in not training mode!')

  def normalize_data(self):
    for k in self._data.keys():
      tmp_data = self._data[k]
      tmp_data = tmp_data - self._norm_stats['mean']
      tmp_data = np.divide(tmp_data, self._norm_stats['std'])
      self._data[k] = tmp_data

  def load_data(self):
    seq_files = self.read_fold_file(self._fold_file)
    self._data = {}
    all_dataset = []
    seq_lens = []

    for sequence_file in tqdm.tqdm(seq_files):
      sequence_file = os.path.join(self._params['data_path'], 
          'nturgb+d_skeletons', sequence_file)

      # the sequence key contains
      # (setup_id, camera_id, subject_id, replication_id, activity_id)
      # sequence shape [num_frames, 25, 3]
      action_sequence, seq_key = read_sequence_kinect_skeletons(sequence_file)
      action_sequence = select_sequence_based_var(action_sequence)
      # sequence shape [num_frames, 16, 3]
      action_sequence = action_sequence[:, _MAJOR_JOINTS, :]

      # Only consider sequences with more than _MIN_REQ_FRAMES frames
      if action_sequence.shape[0]<_MIN_REQ_FRAMES:
        continue

      # center joints in the spine of the skeleton
      root_sequence =  np.expand_dims(action_sequence[:, _SPINE_ROOT, :], axis=1)
      action_sequence = action_sequence - root_sequence
      T, N, D = action_sequence.shape
      seq_lens.append(T)
      # total_frames x n_joints*3
      self._data[seq_key] = action_sequence.reshape((T, -1))
      all_dataset.append(action_sequence)

    all_dataset = np.concatenate(all_dataset, axis=0)
    self.load_compute_norm_stats(all_dataset)
    self.normalize_data()

    self._pose_dim = self._norm_stats['std'].shape[-1]
    self._data_dim = self._pose_dim

    self._data_keys = list(self._data.keys())
    thisname = self.__class__.__name__
    print('[INFO] ({}) The min seq len for mode: {} is: {}'.format(
        thisname, self._mode, min(seq_lens)))
    print('[INFO] ({}) Pose dim: {} Data dim: {}'.format(
        thisname, self._pose_dim, self._data_dim))

  def __len__(self):
    if self._mode == 'train':
      return max(len(self._data_keys), self._params['virtual_dataset_size'])
    return len(self._data_keys)

  def __getitem__(self, idx):
    return self._get_item_train(idx)

  def _get_item_train(self, idx):
    """Get item for the training mode."""
    if self._mode == 'train':
      idx = np.random.choice(len(self._data_keys), 1)[0]

    the_key = self._data_keys[idx]
    # the action id in the files come in 1 based index 
    action_id = the_key[-1] - 1
    source_seq_len = self._params['source_seq_len']
    target_seq_len = self._params['target_seq_len']
    input_size = self._pose_dim
    pose_size = self._pose_dim
    total_frames = source_seq_len + target_seq_len
    src_seq_len = source_seq_len - 1

    encoder_inputs = np.zeros((src_seq_len, input_size), dtype=np.float32)
    decoder_inputs = np.zeros((target_seq_len, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((target_seq_len, pose_size), dtype=np.float32)

    N, _ = self._data[the_key].shape
    start_frame = np.random.randint(0, N-total_frames)
    # total_framesxn_joints*joint_dim
    data_sel = self._data[the_key][start_frame:(start_frame+total_frames), :]

    encoder_inputs[:, 0:input_size] = data_sel[0:src_seq_len,:]
    decoder_inputs[:, 0:input_size] = \
        data_sel[src_seq_len:src_seq_len+target_seq_len, :]
    decoder_outputs[:, 0:pose_size] = data_sel[source_seq_len:, 0:pose_size]

    if self._params['pad_decoder_inputs']:
      query = decoder_inputs[0:1, :]
      decoder_inputs = np.repeat(query, target_seq_len, axis=0)

    return {
        'encoder_inputs': encoder_inputs, 
        'decoder_inputs': decoder_inputs, 
        'decoder_outputs': decoder_outputs,
        'action_id': action_id,
        'action_str': self._action_str[action_id],
    }

  def unormalize_sequence(self, action_sequence):
    sequence = action_sequence*self._norm_stats['std']
    sequence = sequence + self._norm_stats['mean']

    return sequence 

def dataset_factory(params):
  """Defines the datasets that will be used for training and validation."""
  params['num_activities'] = _TOTAL_ACTIONS
  params['virtual_dataset_size'] = params['steps_per_epoch']*params['batch_size']
  params['n_joints'] = _NMAJOR_JOINTS

  eval_mode = 'test' if 'test_phase' in params.keys() else 'eval'
  if eval_mode == 'test':
    train_dataset_fn = None
  else:
    train_dataset = NTURGDDatasetSkeleton(params, mode='train')
    train_dataset_fn = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True
    )

  eval_dataset = NTURGDDatasetSkeleton(
      params, 
      mode=eval_mode
  )
  eval_dataset_fn = torch.utils.data.DataLoader(
      eval_dataset,
      batch_size=1,
      shuffle=True,
      num_workers=1,
      drop_last=True,
      collate_fn=collate_fn,
  ) 

  return train_dataset_fn, eval_dataset_fn



if __name__ == '__main__':
  parser = argparse.ArgumentParser()                                             
  parser.add_argument('--data_path', type=str, default=None)
  parser.add_argument('--pad_decoder_inputs',  action='store_true')
  parser.add_argument('--source_seq_len', type=int, default=40)
  parser.add_argument('--target_seq_len', type=int, default=20)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--steps_per_epoch', type=int, default=200)
  args = parser.parse_args()

  params = vars(args)

  train_dataset_load, val_dataset_load = dataset_factory(params)

  for n, sample in enumerate(val_dataset_load):
    print(n,
          sample['encoder_inputs'].size(),
          sample['decoder_inputs'].size(),
          sample['decoder_outputs'].size(),
          sample['action_ids'].size())
    

#  save_fold_files(
#      'ntu_rgbd/nturgb+d_skeletons', 
#      'ntu_rgbd', 
#      'ntu_rgbd/missing_skeletons.txt'
#  )


