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

"""Visualization of the attention weights."""

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import os
import tqdm
import json
from sklearn.metrics import confusion_matrix

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import data.H36MDataset_v2 as H36M_v2
import models.PoseTransformer as PoseTransformer
import models.PoseEncoderDecoder as PoseEncoderDecoder

import matplotlib.pyplot as plt
import matplotlib


def plot_conf_mat(matrix, action_labels):
  fig, ax = plt.subplots()
  im = ax.imshow(matrix, cmap='Wistia')

  ax.set_xticks(np.arange(len(action_labels)))
  ax.set_yticks(np.arange(len(action_labels)))

  ax.set_xticklabels(action_labels, fontdict={'fontsize':10}, rotation=90)
  ax.set_yticklabels(action_labels, fontdict={'fontsize':10})

#  cbar_kw={}
#  if set_colorbar:
#      cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#      cbar.ax.set_ylabel("", rotation=-90, va="bottom")
#  nmax= np.max(matrix)/2.

  ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

  for i in range(len(action_labels)):
    for j in range(len(action_labels)):
      # color= "w" if round(matrix[i, j],2) < nmax else "black"
      text = ax.text(j, i, round(matrix[i, j], 2),
          ha="center", va="center", color="black", fontsize=5)

  plt.ylabel("")
  plt.xlabel("")
  # ax.set_title("Small plot")
  fig.tight_layout()
  plt.show()
  #plt.savefig(name)
  plt.close()


the_keys = [(5, 'directions', 1), (5, 'directions', 2), (5, 'discussion', 1), (5, 'discussion', 2), (5, 'eating', 1), (5, 'eating', 2), (5, 'greeting', 1), (5, 'greeting', 2), (5, 'phoning', 1), (5, 'phoning', 2), (5, 'posing', 1), (5, 'posing', 2), (5, 'purchases', 1), (5, 'purchases', 2), (5, 'sitting', 1), (5, 'sitting', 2), (5, 'sittingdown', 1), (5, 'sittingdown', 2), (5, 'smoking', 1), (5, 'smoking', 2), (5, 'takingphoto', 1), (5, 'takingphoto', 2), (5, 'waiting', 1), (5, 'waiting', 2), (5, 'walking', 1), (5, 'walking', 2), (5, 'walkingdog', 1), (5, 'walkingdog', 2), (5, 'walkingtogether', 1), (5, 'walkingtogether', 2)
]


def get_windows(
    data, 
    source_seq_len, 
    target_seq_len, 
    pad_decoder_inputs, 
    input_size, n_windows):
  N, _ = data.shape
  src_seq_len = source_seq_len - 1

  encoder_inputs_ = []
  decoder_inputs_ = []
  decoder_outputs_ = []

  start_frame = 0
  for n in range(n_windows):
    encoder_inputs = np.zeros((src_seq_len, input_size), dtype=np.float32)
    decoder_inputs = np.zeros((target_seq_len, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((target_seq_len, input_size), dtype=np.float32)

    # total_frames x n_joints*joint_dim
    total_frames = source_seq_len + target_seq_len
    data_sel = data[start_frame:(start_frame+total_frames), :]
                                                                                 
    encoder_inputs[:, 0:input_size] = data_sel[0:src_seq_len,:]
    decoder_inputs[:, 0:input_size] = data_sel[src_seq_len:src_seq_len+target_seq_len, :]
    decoder_outputs[:, 0:input_size] = data_sel[source_seq_len:, 0:input_size]

    if pad_decoder_inputs:
      query = decoder_inputs[0:1, :]                                             
      decoder_inputs = np.repeat(query, target_seq_len, axis=0)

    encoder_inputs_.append(encoder_inputs)
    decoder_inputs_.append(decoder_inputs)
    decoder_outputs_.append(decoder_outputs)
    start_frame = start_frame + src_seq_len

  return (
      torch.from_numpy(np.stack(encoder_inputs_)),
      torch.from_numpy(np.stack(decoder_inputs_)),
      torch.from_numpy(np.stack(decoder_outputs_))
  )
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--params_json', type=str, default=None)
  parser.add_argument('--model', type=str, default= None)
  args = parser.parse_args()

  _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  params = json.load(open(args.params_json))

  train_dataset_fn, eval_dataset_fn = H36M_v2.dataset_factory(params)
  pose_encoder_fn, pose_decoder_fn = \
      PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)

  potr = PoseTransformer.model_factory(
      params, pose_encoder_fn, pose_decoder_fn)
  potr.load_state_dict(torch.load(args.model, map_location=_DEVICE))
  potr.to(_DEVICE)
  potr.eval()

  all_pred, all_gt = [], []
  n_windows = 8

  the_keys_ = [the_keys[i] for i in range(1, len(the_keys), 2)]
  with torch.no_grad():
    for i in range(len(the_keys_)):
      entry_key = the_keys_[i]  # (5, 'walking', 1)
      data = eval_dataset_fn.dataset._data[entry_key]

      encoder_inputs, decoder_inputs, decoder_outputs = get_windows(
          data, 
          params['source_seq_len'], 
          params['target_seq_len'], 
          params['pad_decoder_inputs'], 
          params['input_dim'], 
          n_windows
      )
      pred_sequence, attn_weights, enc_weights= potr(
          encoder_inputs.to(_DEVICE), 
          decoder_inputs.to(_DEVICE), 
          get_attn_weights=True
      )

      enc_weights = enc_weights.cpu().numpy()
      attn_weights = attn_weights[-1].cpu().numpy()
      attn_weights = [attn_weights[j] for j in range(n_windows)]
      mat = np.concatenate(attn_weights, axis=-1)
      mat = np.concatenate([enc_weights[j] for j in range(n_windows)], axis=-1)

      print(enc_weights.shape)

      fig, ax = plt.subplots(figsize=(20,10))
      ax.matshow(mat)
      plt.ylabel("")
      plt.xlabel("")
      fig.tight_layout()
      #plt.show()
      name = 'vis_attn/%s_.png'%(entry_key[1])
      plt.savefig(name)
      plt.close()




