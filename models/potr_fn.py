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

"""Model function to deploy POTR models for visualization and generation."""


import numpy as np
import os
import sys
import argparse
import json
import time
import cv2

from matplotlib import image
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import training.seq2seq_model_fn as seq2seq_model_fn
import models.PoseTransformer as PoseTransformer
import models.PoseEncoderDecoder as PoseEncoderDecoder
import data.H36MDataset_v2 as H36MDataset_v2
import data.AMASSDataset as AMASSDataset
import utils.utils as utils
import radam.radam as radam
import training.transformer_model_fn as tr_fn
import tqdm

# _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_DEVICE = torch.device('cpu')


def plot_conf_mat(matrix):
  import matplotlib.pyplot as plt
  import matplotlib
  fig, ax = plt.subplots(figsize=(30,30))
  #im = ax.imshow(matrix, cmap='Wistia')
  im = ax.imshow(matrix, cmap='Blues')

  action_labels = ['A%02d'%i for i in range(1, 61, 1)]
  ax.set_xticks(np.arange(len(action_labels)))
  ax.set_yticks(np.arange(len(action_labels)))

  ax.set_xticklabels(action_labels, fontdict={'fontsize':15})#, rotation=90)
  ax.set_yticklabels(action_labels, fontdict={'fontsize':15})

  ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

  for i in range(len(action_labels)):
    for j in range(len(action_labels)):
      # color= "w" if round(matrix[i, j],2) < nmax else "black"
      text = ax.text(j, i, round(matrix[i, j], 2),
          ha="center", va="center", color="black", fontsize=10)

  plt.ylabel("")
  plt.xlabel("")
  # ax.set_title("Small plot")
  fig.tight_layout()
  #plt.show()
  plt.savefig('confusion_matrix.png')
  plt.close()

def crop_image(img):
  size = max(img.shape[0], img.shape[1])
  h = int(size*0.30)
  w = int(size*0.30)
  cy = img.shape[0]//2
  cx = img.shape[1]//2

  crop = img[cy-h//2:cy+h//2, cx-w//2:cx+w//2]
  return crop



def visualize_h36mdataset():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file', type=str)
  parser.add_argument('--model_file', type=str)
  parser.add_argument('--data_path', type=str, default=None)
  args = parser.parse_args()

  params = json.load(open(args.config_file))
  if args.data_path is not None:
    params['data_path'] = args.data_path
  args.data_path = params['data_path']
  train_dataset_fn, eval_dataset_fn = tr_fn.dataset_factory(params)

  pose_encoder_fn, pose_decoder_fn = \
      PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)

  for k,v in params.items():
    print('[INFO] (POTRFn@main) {}: {}'.format(k, v))


  # ids of most common actions in H36M
  actions = [('walking', 12),  ('eating', 2),  ('smoking', 9),  
      ('discussion', 1), ('directions', 0)]

  with torch.no_grad():
    for i in range(len(actions)):
      action, acidx = actions[i]
      sample = next(iter(eval_dataset_fn))

      enc_inputs = sample['encoder_inputs'].to(_DEVICE)
      dec_inputs = sample['decoder_inputs'].to(_DEVICE)
      gts = np.squeeze(sample['decoder_outputs'].cpu().numpy())[8*acidx:8*acidx+8]
      ins = np.squeeze(sample['encoder_inputs'].cpu().numpy())[8*acidx:8*acidx+8]

      ins = eval_dataset_fn.dataset.unnormalize_pad_data_to_expmap(ins)
      H36MDataset_v2.visualize_sequence(
        ins[0:1], args.data_path, prefix='skeletons/%s/gt_in'%action, colors=['gray', 'gray'])

      #print(gts.shape)
      gts = eval_dataset_fn.dataset.unnormalize_pad_data_to_expmap(gts)
      H36MDataset_v2.visualize_sequence(
          gts[0:1], args.data_path, prefix='skeletons/%s/gt'%action, colors=['gray', 'gray'])

      enc_inputs = torch.squeeze(enc_inputs)
      dec_inputs = torch.squeeze(dec_inputs)

      model = PoseTransformer.model_factory(
          params, 
          pose_encoder_fn, 
          pose_decoder_fn
      )

      model.load_state_dict(torch.load(args.model_file, map_location=_DEVICE))
      model.to(_DEVICE)
      model.eval()

      prediction, attn_weights, memory = model(
          enc_inputs, 
          dec_inputs, 
          get_attn_weights=True
      )

      prediction = prediction[-1][8*acidx:8*acidx+8].cpu().numpy()[0:1]
      preds = eval_dataset_fn.dataset.unnormalize_pad_data_to_expmap(prediction)
      H36MDataset_v2.visualize_sequence(preds, args.data_path, 
          prefix='skeletons/%s/pred'%action, colors=['red', 'red'])


def compute_mean_average_precision(prediction, target, dataset_fn):
  pred = prediction.cpu().numpy().squeeze()
  tgt = target.cpu().numpy().squeeze()

  T, D = pred.shape

  pred = dataset_fn.dataset.unormalize_sequence(pred)
  tgt = dataset_fn.dataset.unormalize_sequence(tgt)

  pred = pred.reshape((T, -1, 3))
  tgt = tgt.reshape((T, -1, 3))

  mAP, _, _, (TP, FN) = utils.compute_mean_average_precision(
      pred, tgt, seq2seq_model_fn._MAP_TRESH, per_frame=True
  )

  return mAP, TP, FN

def compute_mpjpe(prediction, target, dataset_fn):
  pred = prediction.cpu().numpy().squeeze()
  tgt = target.cpu().numpy().squeeze()

  T, D = pred.shape

  pred = dataset_fn.dataset.unormalize_sequence(pred)
  tgt = dataset_fn.dataset.unormalize_sequence(tgt)

  pred = pred.reshape((T, -1, 3))
  tgt = tgt.reshape((T, -1, 3))

  # seq_len x n_joints
  norm = np.squeeze(np.linalg.norm(pred-tgt, axis=-1))
  return norm

def compute_test_mAP_nturgbd():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file', type=str)
  parser.add_argument('--model_file', type=str)
  parser.add_argument('--data_path', type=str, default=None)
  args = parser.parse_args()

  params = json.load(open(args.config_file))
  if args.data_path is not None:
    params['data_path'] = args.data_path
  args.data_path = params['data_path']
  params['test_phase'] = True

  _, test_dataset_fn = tr_fn.dataset_factory(params)
  pose_encoder_fn, pose_decoder_fn = \
      PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)

  for k,v in params.items():
    print('[INFO] (POTRFn@main) {}: {}'.format(k, v))

  model = PoseTransformer.model_factory(
      params, 
      pose_encoder_fn, 
      pose_decoder_fn
  )

  model.load_state_dict(torch.load(args.model_file, map_location=_DEVICE))
  model.to(_DEVICE)
  model.eval()

  FN = np.zeros((params['target_seq_len'],), dtype=np.float32)
  TP = np.zeros((params['target_seq_len'],), dtype=np.float32)
  FN_joint = np.zeros((params['n_joints'],), dtype=np.float32)
  TP_joint = np.zeros((params['n_joints'],), dtype=np.float32)
  MPJPE = np.zeros((params['n_joints'],), dtype=np.float32)

  pred_activity = []
  gt_activity = []

  with torch.no_grad():
    print('Running testing...')
    for n, sample in tqdm.tqdm(enumerate(test_dataset_fn)):

      enc_inputs = sample['encoder_inputs'].to(_DEVICE)
      dec_inputs = sample['decoder_inputs'].to(_DEVICE)
      gts = sample['decoder_outputs'].to(_DEVICE)

  #    ins = np.squeeze(sample['encoder_inputs'].cpu().numpy())


      outputs = model(
          enc_inputs, 
          dec_inputs, 
          get_attn_weights=True
      )

      if params['predict_activity']:
        a_ids = sample['action_ids']
        prediction, out_logits, attn_weights, memory = outputs
        out_class = torch.argmax(out_logits[-1].softmax(-1), -1)
      else:
        prediction, attn_weights, memory = outputs

      
      mAP, TP_, FN_ = compute_mean_average_precision(prediction[-1], gts, test_dataset_fn)
      MPJPE_ = compute_mpjpe(prediction[-1], gts, test_dataset_fn)

      # reduce by frame to get per joint MPJPE
      MPJPE = MPJPE + np.sum(MPJPE_, axis=0)

      # reduce by frame to get per joint AP
      TP_joint = TP_joint + np.sum(TP_, axis=0)
      FN_joint = FN_joint + np.sum(FN_, axis=0)

      # reduce by joint to get per frame AP
      TP_ = np.sum(TP_, axis=-1)
      FN_ = np.sum(FN_, axis=-1)
      TP = TP + TP_
      FN = FN + FN_


      # print(n, ':', prediction[-1].size(), out_class.item(), a_ids.item(), mAP, TP.shape, FN.shape)

      if params['predict_activity']:
        pred_activity.append(out_class.item())
        gt_activity.append(a_ids.item())

  #accurracy = (np.array(gt_activity) == np.array(pred_activity)).astype(np.float32).sum()
  #accurracy = accurracy / len(gt_activity)
  accurracy = -1
  if params['predict_activity']:
    accurracy = accuracy_score(gt_activity, pred_activity, normalize='true')
    conf_matrix = confusion_matrix(gt_activity, pred_activity, normalize='true')
    plot_conf_mat(conf_matrix)

  AP = TP / (TP+FN)
  AP_joints = TP_joint / (TP_joint + FN_joint)
  MPJPE = MPJPE / (n*params['target_seq_len'])

  print('[INFO] The mAP per joint\n', np.around(AP_joints, 2))
  print('[INFO] The MPJPE\n', np.around(MPJPE,4)*100.0)

  print('[INFO] The accuracy: {} mAP: {}'.format(
      round(accurracy, 2), round(np.mean(AP), 2)))

  ms_range = [0.08, 0.160, 0.320, 0.400, 0.5, 0.66]
  FPS = 30.0
  ms_map = []
  for ms in ms_range:
    nf = int(round(ms*FPS))
    ms_map.append(np.mean(AP[0:nf]))

  print()
  print("{0: <16} |".format("milliseconds"), end="")
  for ms in ms_range:
    print(" {0:5d} |".format(int(ms*1000)), end="")
  print()

  print("{0: <16} |".format("global mAP"), end="")
  for mAP in ms_map:
    print(" {0:.3f} |".format(mAP), end="")

  print()


def visualize_attn_weights():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file', type=str)
  parser.add_argument('--model_file', type=str)
  parser.add_argument('--data_path', type=str, default=None)
  args = parser.parse_args()

  params = json.load(open(args.config_file))
  if args.data_path is not None:
    params['data_path'] = args.data_path
  args.data_path = params['data_path']
  train_dataset_fn, eval_dataset_fn = tr_fn.dataset_factory(params)

  pose_encoder_fn, pose_decoder_fn = \
      PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)

  model = PoseTransformer.model_factory(
      params, 
      pose_encoder_fn, 
      pose_decoder_fn
  )

  model.load_state_dict(torch.load(args.model_file, map_location=_DEVICE))
  model.to(_DEVICE)
  model.eval()

  for k,v in params.items():
    print('[INFO] (POTRFn@main) {}: {}'.format(k, v))

  # ids of most common actions in H36M
  actions = [('walking', 12)]
  #[('walking', 12),  ('eating', 2),  ('smoking', 9),  
#      ('discussion', 1), ('directions', 0)]

  with torch.no_grad():
    for i in range(len(actions)):
      action, acidx = actions[i]
      sample = next(iter(eval_dataset_fn))

      enc_inputs = sample['encoder_inputs'].to(_DEVICE)
      dec_inputs = sample['decoder_inputs'].to(_DEVICE)

      enc_inputs = torch.squeeze(enc_inputs)
      dec_inputs = torch.squeeze(dec_inputs)

      prediction, attn_weights, enc_weights = model(
          enc_inputs, 
          dec_inputs, 
          get_attn_weights=True
      )

    attn_weights= attn_weights[-1][8*acidx:8*acidx+8].cpu().numpy()[0:1]
    attn_weights = np.squeeze(attn_weights)
    print(attn_weights.shape)
    path = 'skeletons/%s'%action
    in_imgs_ = [crop_image(cv2.imread(os.path.join(path, x)) )
        for x in os.listdir(path) if 'gt_in' in x]

    in_imgs = [in_imgs_[i] for i in range(0, len(in_imgs_), 2)]

    pred_imgs = [crop_image(cv2.imread(os.path.join(path, x)))
        for x in os.listdir(path) if 'pred_0' in x]

    the_shape = in_imgs[0].shape
    cx = the_shape[1]//2
    cy = the_shape[0]//2

    in_imgs = np.concatenate(in_imgs, axis=1)
    pred_imgs = np.concatenate(pred_imgs, axis=1)

    #cv2.imshow('In IMG', in_imgs)
    #cv2.imshow('pred IMG', pred_imgs)
    #cv2.waitKey()

    spaces_between = 5
    print(in_imgs.shape, pred_imgs.shape, the_shape)
    canvas = np.ones(
        (in_imgs.shape[0]*spaces_between, in_imgs.shape[1], 3), 
        dtype=in_imgs.dtype)*255

    canvas[0:the_shape[0], :] = in_imgs
    canvas[the_shape[0]*(spaces_between-1):, 0:pred_imgs.shape[1]] = pred_imgs

    #cx_pred = cx + the_shape[1]*(spaces_between-1) - cx//2
    cy_pred = cy + the_shape[0]*(spaces_between-1) - cy//3*2
    print(attn_weights.min(), attn_weights.max())

    mean = attn_weights.mean()
    #plt.imshow(canvas, origin='lower')
    pil_canvas = Image.fromarray(canvas)
    d_canvas = ImageDraw.Draw(pil_canvas)

    for pred_idx in range(attn_weights.shape[0]):
      # cy_pred = cy + pred_idx*the_shape[0]
      cx_pred = cx + pred_idx*the_shape[1]
      #cv2.circle(canvas, (cx_pred, cy_pred), 5, [0,255,0],  -1)
      for ii, in_idx in enumerate(range(0, attn_weights.shape[1], 2)):
        # cy_in = cy + ii*the_shape[0]
        cx_in = cx + ii*the_shape[1]
        this_weight = attn_weights[pred_idx, in_idx]
        if this_weight > mean:
          #d_canvas.line([(cx+cx//2, cy_in), (cx_pred, cy_pred)], fill=(255,0,0, 25), width=this_weight/mean)
          d_canvas.line([(cx_in, cy+cy//3*2), (cx_pred, cy_pred)], fill=(255,0,0, 25), width=this_weight/mean)

    name = 'the_canvas.png'
    #cv2.imwrite('the_canvas.jpg', canvas)
    # plt.show()
    #plt.imsave(name, canvas)
    pil_canvas.save(name)
    print(pil_canvas.info)

    fig, ax = plt.subplots(figsize=(20,10))
    ax.matshow(attn_weights)
    plt.ylabel("")
    plt.xlabel("")
    fig.tight_layout()
    #plt.show()
    name = 'attn_map.png'
    plt.savefig(name)
    plt.close()


if __name__ == '__main__':
  # visualize_h36mdataset()
  visualize_attn_weights()
  #compute_test_mAP_nturgbd()








