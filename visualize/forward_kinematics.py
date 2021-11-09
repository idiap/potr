"""Visualize predictions as a sequence of skeletons."""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anumation
import numpy as np
import json
import argparse
import viz
import os
import sys
import h5py

sys.path.append('../')

import utils.utils as utils




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_path', type=str)
  parser.add_argument('--input_sample', type=str)

  args = parser.parse_args()

  parent, offset, rot_ind, exp_map_ind = utils.load_constants(args.dataset_path)

  print(parent.shape)  
  print(rot_ind)
  print(offset)

  # expmap = np.load(args.input_sample)
  with h5py.File(args.input_sample, 'r') as h5f:
    expmap_gt = h5f['expmap/gt/walking_0'][:]
    expmap_pred = h5f['expmap/preds/walking_0'][:]

  nframes_gt, nframes_pred = expmap_gt.shape[0], expmap_pred.shape[0]
  input_pose = np.vstack((expmap_gt, expmap_pred))
  print(input_pose.shape)
  expmap_all = utils.revert_coordinate_space(input_pose, np.eye(3), np.zeros(3))

  print(expmap_gt.shape, expmap_pred.shape)

  expmap_gt = expmap_all[:nframes_gt,:]
  expmap_pred = expmap_all[nframes_gt,:]
  # compute 3d points for each frame
  xyz_gt, xyz_pred = np.zeros((nframes_gt, 96)), np.zeros((nframes_pred, 96))

  print(xyz_gt.shape, xyz_pred.shape)

  for i in range(nframes_gt):
    xyz_gt[i, :] = utils.compute_forward_kinematics(
        expmap_gt[i, :], parent, offset, rot_ind, exp_map_ind)

  #for i in range(nframes_pred):
  #  xyz_pred[i, :] = compute_forward_kinematics(
  #      expmap_pred[i, :], parent, offset, rot_ind, exp_map_ind)
  
  # plot and animate the poses
  fig = plt.figure()
  ax = plt.gca(projection='3d')
  ob = viz.Ax3DPose(ax)

  for i in range(nframes_gt):
    ob.update(xyz_gt[i, :])
    plt.show(block=False)
    fig.canvas.draw()
    plt.pause(0.01)





