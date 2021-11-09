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

"""Set of utility functions."""


import torch
import numpy as np
import copy
import json
import os
import cv2

import torch.nn as nn

def expmap_to_euler(action_sequence):
  rotmats = expmap_to_rotmat(action_sequence)
  eulers = rotmat_to_euler(rotmats)
  return eulers

def expmap_to_rotmat(action_sequence):
  """Convert exponential maps to rotmats.

  Args:
    action_sequence: [n_samples, n_joints, 3]
  Returns:
    Rotation matrices for exponenital maps [n_samples, n_joints, 9].
  """
  n_samples, n_joints, _ = action_sequence.shape
  expmap = np.reshape(action_sequence, [n_samples*n_joints, 1, 3])
  # first three values are positions, so technically it's meaningless to convert them,
  # but we do it anyway because later we discard this values anywho
  rotmats = np.zeros([n_samples*n_joints, 3, 3])
  for i in range(rotmats.shape[0]):
    rotmats[i] = cv2.Rodrigues(expmap[i])[0]
  rotmats = np.reshape(rotmats, [n_samples, n_joints, 3*3])
  return rotmats

def rotmat_to_expmap(action_sequence):
  """Convert rotmats to expmap.

  Args:
    action_sequence: [n_samples, n_joints, 9]
  Returns:
    Rotation exponenital maps [n_samples, n_joints, 3].
  """
  n_samples, n_joints, _ = action_sequence.shape
  rotmats = np.reshape(action_sequence, [n_samples*n_joints, 3, 3])
  # first three values are positions, so technically it's meaningless to convert them,
  # but we do it anyway because later we discard this values anywho
  expmaps = np.zeros([n_samples*n_joints, 3, 1])
  for i in range(rotmats.shape[0]):
    expmaps[i] = cv2.Rodrigues(rotmats[i])[0]
  expmaps = np.reshape(expmaps, [n_samples, n_joints, 3])

  return expmaps 


def rotmat_to_euler(action_sequence):
  """Convert exponential maps to rotmats.

  Args:
    action_sequence: [n_samples, n_joints, 9]
  Returns:
    Euler angles for rotation maps given [n_samples, n_joints, 3].
  """
  n_samples, n_joints, _ = action_sequence.shape
  rotmats = np.reshape(action_sequence, [n_samples*n_joints, 3, 3])
  eulers = np.zeros([n_samples*n_joints, 3])
  for i in range(eulers.shape[0]):
    eulers[i] = rotmat2euler(rotmats[i])
  eulers = np.reshape(eulers, [n_samples, n_joints, 3])
  return eulers


def rotmat2euler(R):
  """Converts a rotation matrix to Euler angles.
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

  Args:
    R: a 3x3 rotation matrix

  Returns:
    eul: a 3x1 Euler angle representation of R
  """
  if R[0,2] >= 1 or R[0,2] <= -1:
    # special case values are out of bounds for arcsinc
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;
  else:
    E2 = -np.arcsin(R[0,2])
    E1 = np.arctan2(R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2(R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul


def load_constants(data_path):
  offset = json.load(open(os.path.join(data_path, 'offset.json')))
  parent = json.load(open(os.path.join(data_path, 'parent.json')))
  rot_ind = json.load(open(os.path.join(data_path, 'rot_ind.json')))

  parent = np.array(parent)-1
  offset = np.array(offset).reshape(-1, 3)
  exp_map_ind = np.split(np.arange(4, 100)-1, 32)

  return parent, offset, rot_ind, exp_map_ind


def compute_forward_kinematics(angles, parent, offset, rotInd, expmapInd):
  """Computes forward kinematics from angles to 3d points.

  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

  Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """
  assert len(angles) == 99, 'Incorrect number of angles.'

  # Structure that indicates parents for each joint
  njoints = 32
  xyzStruct = [dict() for x in range(njoints)]

  for i in np.arange(njoints):
    if not rotInd[i] : # If the list is empty
      xangle, yangle, zangle = 0, 0, 0
    else:
      xangle = angles[rotInd[i][0]-1]
      yangle = angles[rotInd[i][1]-1]
      zangle = angles[rotInd[i][2]-1]

    r = angles[expmapInd[i]]
    thisRotation = expmap2rotmat(r)
    thisPosition = np.array([xangle, yangle, zangle])

    if parent[i] == -1: # Root node
      xyzStruct[i]['rotation'] = thisRotation
      xyzStruct[i]['xyz'] = np.reshape(offset[i,:], (1,3)) + thisPosition
    else:
      xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot(
          xyzStruct[parent[i]]['rotation']) + xyzStruct[parent[i]]['xyz']
      xyzStruct[i]['rotation'] = thisRotation.dot(
          xyzStruct[parent[i]]['rotation'])

  xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
  xyz = np.array(xyz).squeeze()
  xyz = xyz[:,[0,2,1]]
  # xyz = xyz[:,[2,0,1]]

  return np.reshape( xyz, [-1] )


def revert_coordinate_space(channels, R0, T0):
  """Arrange poses to a canonical form to face the camera.

  Bring a series of poses to a canonical form so they are facing the camera 
  when they start. Adapted from
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

  Args:
    channels: n-by-99 matrix of poses
    R0: 3x3 rotation for the first frame
    T0: 1x3 position for the first frame

  Returns:
    channels_rec: The passed poses, but the first has T0 and R0, and the 
    rest of the sequence is modified accordingly.
  """
  n, d = channels.shape

  channels_rec = copy.copy(channels)
  R_prev = R0
  T_prev = T0
  rootRotInd = np.arange(3,6)

  for ii in range(n):
    R_diff = expmap2rotmat(channels[ii, rootRotInd])
    R = R_diff.dot(R_prev)

    channels_rec[ii, rootRotInd] = rotmat2expmap(R)
    T = T_prev + (R_prev.T).dot(np.reshape(channels[ii,:3],[3,1])).reshape(-1)
    channels_rec[ii,:3] = T
    T_prev = T
    R_prev = R

  return channels_rec


def rotmat2quat(R):
  """Converts a rotation matrix to a quaternion.

  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

  Args:
    R: 3x3 rotation matrix

  Returns:
    q: 1x4 quaternion
  """
  rotdiff = R - R.T;

  r = np.zeros(3)
  r[0] = -rotdiff[1,2]
  r[1] =  rotdiff[0,2]
  r[2] = -rotdiff[0,1]
  sintheta = np.linalg.norm(r) / 2;
  r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );

  costheta = (np.trace(R)-1) / 2;

  theta = np.arctan2( sintheta, costheta );

  q      = np.zeros(4)
  q[0]   = np.cos(theta/2)
  q[1:] = r0*np.sin(theta/2)
  return q


def quat2expmap(q):
  """Convert quaternions to an exponential map.

  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

  Args:
    q: 1x4 quaternion

  Returns:
    r: 1x3 exponential map

  Raises:
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  if (np.abs(np.linalg.norm(q)-1)>1e-3):
    raise(ValueError, "quat2expmap: input quaternion is not norm 1")

  sinhalftheta = np.linalg.norm(q[1:])
  coshalftheta = q[0]

  r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
  theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
  theta = np.mod( theta + 2*np.pi, 2*np.pi )

  if theta > np.pi:
    theta =  2 * np.pi - theta
    r0    = -r0

  r = r0 * theta
  return r


def rotmat2expmap(R):
  return quat2expmap( rotmat2quat(R) )


def expmap2rotmat(r):
  """Converts an exponential map (axis angle number) to rotation matrix.

  Converts an exponential map angle to a rotation matrix Matlab port to python 
  for evaluation purposes. This is also called Rodrigues' formula and can be
  found also implemented in opencv as cv2.Rodrigues.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

  Args:
    r: 1x3 exponential map

  Returns:
    R: 3x3 rotation matrix
  """
  theta = np.linalg.norm( r )
  r0  = np.divide( r, theta + np.finfo(np.float32).eps )
  r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
  r0x = r0x - r0x.T
  R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);

  return R


def revert_output_format(
    poses,
    data_mean,
    data_std,
    dim_to_ignore,
    actions,
    use_one_hot):
  """Transforms pose predictions to a more interpretable format.

  Converts the output of the neural network to a format that is more easy to
  manipulate for, e.g. conversion to other format or visualization

  Args:
    poses: Sequence of pose predictions. A list with (seq_length) entries,
      each with a (batch_size, dim) output

  Returns:
    poses_out: List of tensors each of size (batch_size, seq_length, dim).
  """
  seq_len = len(poses)
  if seq_len == 0:
    return []

  batch_size, dim = poses[0].shape

  poses_out = np.concatenate(poses)
  poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
  poses_out = np.transpose(poses_out, [1, 0, 2])

  poses_out_list = []
  for i in range(poses_out.shape[0]):
    poses_out_list.append(
      unnormalize_data(poses_out[i, :, :], data_mean, data_std, 
                      dim_to_ignore, actions, use_one_hot))

  return poses_out_list


def unnormalize_data(
    normalizedData, 
    data_mean, 
    data_std, 
    dimensions_to_ignore=None,
    actions=[], 
    use_one_hot=False):
  """
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

  Args
    normalizedData: nxd matrix with normalized data
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    actions: list of strings with the encoded actions
    use_one_hot: whether the data comes with one-hot encoding
  Returns
    origData: data originally used to
  """
  T = normalizedData.shape[0]
  D = data_mean.shape[0]

  origData = np.zeros((T, D), dtype=np.float32)
  dimensions_to_ignore = [] if dimensions_to_ignore==None else dimensions_to_ignore
  dimensions_to_use = [i for i in range(D) if i not in dimensions_to_ignore]
  dimensions_to_use = np.array(dimensions_to_use)
  #print('Size of the normalized data', normalizedData.shape)
  #print('Size of the mean data', data_mean.shape[0])
  #print('Lenght of the dimensions to use', len(dimensions_to_use))

  if use_one_hot:
    origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
  else:
    # print('++++++++++++++++++++',origData.shape, normalizedData.shape, len(dimensions_to_use))
    origData[:, dimensions_to_use] = normalizedData

  # potentially ineficient, but only done once per experiment
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(origData, stdMat) + meanMat
  return origData


def get_srnn_gts(
    actions,
    model,
    test_set,
    data_mean,
    data_std,
    dim_to_ignore,
    one_hot,
    to_euler=True):
  """
  Get the ground truths for srnn's sequences, and convert to Euler angles.
  (the error is always computed in Euler angles).

  Args
    actions: a list of actions to get ground truths for.
    model: training model we are using (we only use the "get_batch" method).
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map

  Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
  """
  srnn_gts_euler = {}

  for action in actions:

    srnn_gt_euler = []
    _, _, srnn_expmap = model.get_batch_srnn( test_set, action )

    # expmap -> rotmat -> euler
    for i in np.arange( srnn_expmap.shape[0] ):
      denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          for k in np.arange(3,97,3):
            denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

      srnn_gt_euler.append( denormed );

    # Put back in the dictionary
    srnn_gts_euler[action] = srnn_gt_euler

  return srnn_gts_euler



def normal_init_(layer, mean_, sd_, bias, norm_bias=True):
  """Intialization of layers with normal distribution with mean and bias"""
  classname = layer.__class__.__name__
  # Only use the convolutional layers of the module
  #if (classname.find('Conv') != -1 ) or (classname.find('Linear')!=-1):
  if classname.find('Linear') != -1:
    print('[INFO] (normal_init) Initializing layer {}'.format(classname))
    layer.weight.data.normal_(mean_, sd_)
    if norm_bias:
      layer.bias.data.normal_(bias, 0.05)
    else:
      layer.bias.data.fill_(bias)


def weight_init(
    module, 
    mean_=0, 
    sd_=0.004, 
    bias=0.0, 
    norm_bias=False, 
    init_fn_=normal_init_):
  """Initialization of layers with normal distribution"""
  moduleclass = module.__class__.__name__
  try:
    for layer in module:
      if layer.__class__.__name__ == 'Sequential':
        for l in layer:
          init_fn_(l, mean_, sd_, bias, norm_bias)
      else:
        init_fn_(layer, mean_, sd_, bias, norm_bias)
  except TypeError:
    init_fn_(module, mean_, sd_, bias, norm_bias)



def xavier_init_(layer, mean_, sd_, bias, norm_bias=True):
  classname = layer.__class__.__name__
  if classname.find('Linear')!=-1:
    print('[INFO] (xavier_init) Initializing layer {}'.format(classname))
    nn.init.xavier_uniform_(layer.weight.data)
    # nninit.xavier_normal(layer.bias.data)
    if norm_bias:
      layer.bias.data.normal_(0, 0.05)
    else:
      layer.bias.data.zero_()


def create_dir_tree(base_dir):
  dir_tree = ['models', 'tf_logs', 'config', 'std_log']
  for dir_ in dir_tree:
    os.makedirs(os.path.join(base_dir, dir_), exist_ok=True)


def create_look_ahead_mask(seq_length, is_nonautoregressive=False):
  """Generates a binary mask to prevent to use future context in a sequence."""
  if is_nonautoregressive:
    return np.zeros((seq_length, seq_length), dtype=np.float32)
  x = np.ones((seq_length, seq_length), dtype=np.float32)
  mask = np.triu(x, 1).astype(np.float32)
  return mask  # (seq_len, seq_len)


def pose_expmap2rotmat(input_pose):
  """Convert exponential map pose format to rotation matrix pose format."""
  pose_rotmat = []
  for j in np.arange(input_pose.shape[0]):
    rot_mat = [expmap2rotmat(input_pose[j, k:k+3]) for k in range(3, 97, 3)]
    pose_rotmat.append(np.stack(rot_mat).flatten())

  pose_rotmat = np.stack(pose_rotmat)
  return pose_rotmat


def expmap23d_sequence(sequence, norm_stats, params):
  viz_poses = revert_output_format(
      [sequence], norm_stats['mean'], norm_stats['std'],
      norm_stats['dim_to_ignore'], params['action_subset'],
      params['use_one_hot'])

  nframes = sequence.shape[0]
  expmap = revert_coordinate_space(
          viz_poses[0], np.eye(3), np.zeros(3))
  xyz_data = np.zeros((nframes, 96))
  for i in range(nframes):
    xyz_data[i, :] = compute_forward_kinematics(
        expmap[i, :],
        params['parent'], 
        params['offset'], 
        params['rot_ind'], 
        params['exp_map_ind']
    )

  return xyz_data


def get_lr_fn(params, optimizer_fn):
  """Creates the function to be used to generate the learning rate."""
  if params['learning_rate_fn'] == 'step':
    return torch.optim.lr_scheduler.StepLR(
      optimizer_fn, step_size=params['lr_step_size'], gamma=0.1
    )
  elif params['learning_rate_fn'] == 'exponential':
    return torch.optim.lr_scheduler.ExponentialLR(
      optimizer_fn, gamma=0.95
    )
  elif params['learning_rate_fn'] == 'linear':
    # sets learning rate by multipliying initial learning rate times a function
    lr0, T = params['learning_rate'], params['max_epochs']
    lrT = lr0*0.1
    m = (lrT - 1) / T
    lambda_fn =  lambda epoch: m*epoch + 1.0
    return torch.optim.lr_scheduler.LambdaLR(
      optimizer_fn, lr_lambda=lambda_fn
    )
  elif params['learning_rate_fn'] == 'beatles':
    # D^(-0.5)*min(i^(-0.5), i*warmup_steps^(-1.5))
    D = float(params['model_dim'])
    warmup = params['warmup_epochs']
    lambda_fn = lambda e: (D**(-0.5))*min((e+1.0)**(-0.5), (e+1.0)*warmup**(-1.5))
    return torch.optim.lr_scheduler.LambdaLR(
      optimizer_fn, lr_lambda=lambda_fn
    )
  else:
    raise ValueError('Unknown learning rate function: {}'.format(
        params['learning_rate_fn']))


def compute_mean_average_precision(prediction, target, threshold, per_frame=False):
    """
    Args:
      prediction: unormalized sequece of shape [seq_len, num_joints, 3]
      target: unormalized sequence of shape [seq_len, num_joints, 3]
      threshold: float
    """

    # compute the norm for the last axis: (x,y,z) coordinates
    # [num_frames x num_joints]
    TP = np.linalg.norm(prediction-target, axis=-1) <= threshold
    TP_ = TP.astype(int)
    FN_ = np.logical_not(TP).astype(int)

    # [num_joints]
    TP = np.sum(TP_, axis=0)
    FN = np.sum(FN_, axis=0)
    # compute recall for each joint
    recall = TP / (TP+FN)
    # average over joints
    mAP = np.mean(recall)
    if per_frame:
      return mAP, TP, FN, (TP_, FN_)

    return mAP, TP, FN


