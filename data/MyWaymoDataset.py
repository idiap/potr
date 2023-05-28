import numpy as np
import os
import json
import argparse
import logging
import sys
import copy

import torch
import torch.nn.functional as F


class MyWaymoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, source_length, target_length, normalize=True):
        src_hkp_fn = f"inp_hkp_{source_length}.npy"
        tgt_hkp_fn = f"out_hkp_{target_length}.npy"
        src_box_fn = f"inp_box_{source_length}.npy"
        tgt_box_fn = f"out_box_{target_length}.npy"
        self.source_length = source_length
        self.target_length = target_length
        self.normalize = normalize  # Whether we should normalize all points so that the first box center is at the origin
        self.src_hkp = np.load(
            os.path.join(data_dir, src_hkp_fn)
        )  # [N * source_length * 15 * 4]
        self.tgt_hkp = np.load(
            os.path.join(data_dir, tgt_hkp_fn)
        )  # [N * target_length * 15 * 4]
        self.src_box = np.load(
            os.path.join(data_dir, src_box_fn)
        )  # [N * source_length * 3  * 3]
        self.tgt_box = np.load(
            os.path.join(data_dir, tgt_box_fn)
        )  # [N * target_length * 3  * 3]

    def __len__(self):
        return self.src_hkp.shape[0]

    def __getitem__(self, idx):
        hkp_data = np.concatenate(
            (self.src_hkp, self.tgt_hkp), axis=1
        )  # [N * 50 * 15 * 4]
        box_data = np.concatenate(
            (self.src_box, self.tgt_box), axis=1
        )  # [N * 50 * 3  * 3]
        hkp_seq = hkp_data[idx, :, :, :]  # [50 * 15 * 4]
        box_seq = box_data[idx, :, :2, :]  # [50 * 2  * 3] take only position and speed

        # Normalize all positional points by the coordinate of the first box center
        if self.normalize:
            x = box_seq[0, 0, 0]
            y = box_seq[0, 0, 1]
            z = box_seq[0, 0, 2]
            box_seq[:, 0, :] = box_seq[:, 0, :] - [x, y, z]  # [50 * 2 * 3]

            # normalize a keypoint only if it is valid
            hkp_seq_validity = np.expand_dims(hkp_seq[:, :, 3], 2)  # [50 * 15 * 1]
            hkp_seq_validity = np.concatenate(
                (
                    hkp_seq_validity,
                    hkp_seq_validity,
                    hkp_seq_validity,
                    hkp_seq_validity,
                ),
                axis=2,
            )  # [50*15*4]

            hkp_seq = hkp_seq - hkp_seq_validity * [x, y, z, 0]

        return (box_seq, hkp_seq)


if __name__ == "__main__":
    training_dataset = MyWaymoDataset(
        data_dir="/home/yukai/Desktop/potr/data/my_waymo/train/train_10_40",
        source_length=10,
        target_length=40,
        normalize=True,
    )
    train_dataset_fn = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )

    box_seq, hkp_seq = next(iter(train_dataset_fn))
with np.printoptions(precision=3, suppress=True):
    print(f"box sequence shape: {box_seq.shape}, hkp sequence shape: {hkp_seq.shape}")
    print("First sequence: ")
    print(box_seq[0, :, 0, :])
    print(hkp_seq[0, :, 0, :])
