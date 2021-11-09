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

"""Converting txt files of H36M dataset to numpy matrices."""


import numpy as np
import os
import argparse


if __name__ == '__main__':
  parser =  argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default=None)
  parser.add_argument('--output_path', type=str, default=None)
  args = parser.parse_args()


  data_dirs = [os.path.join(args.data_path, x) 
                for x in os.listdir(args.data_path)]

  for d in data_dirs:
    dirs = [os.path.join(d, x) for x in os.listdir(d)]
    dir_path = d.split('/')[-3:]

    out_path = os.path.join(args.output_path, '/'.join(dir_path))
    os.makedirs(out_path, exist_ok=True)
    for dd in dirs:
      print('[ANMG/D] Loading file:', dd)
      txt_mat = np.loadtxt(dd, delimiter=',').astype(np.float32)
      in_file = dd.split('/')[-1].split('.')[0]
      out_file = os.path.join(out_path, in_file+'.npy')
      print('[ANMG/D] Saving to:', out_file)
      np.save(out_file, txt_mat)












