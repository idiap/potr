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

"""Sequence to sequence model for human motion prediction.

The model has been implemented according to [1] and adapted from its pytorch
version [2]. The reimplementation has the purpose of reducing clutter in
code and for learing purposes.

[1]
[2]
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F



class Seq2SeqModel(nn.Module):
  """Sequence to sequence model."""
  def __init__(
      self,
      architecture='tied',
      source_seq_len=50,
      target_seq_len=25,
      rnn_size=1024, # hidden recurrent layer size
      num_layers=1,
      max_gradient_norm=5,
      batch_size=16,
      learning_rate=0.005,
      learning_rate_decay_factor=0.95,
      loss_to_use='sampling_based',
      number_of_actions=1,
      one_hot=True,
      residual_velocities=False,
      dropout=0.0,
      dtype=torch.float32,
      device=None):
    """
    Args:
      architecture: [basic, tied] whether to tie the decoder and decoder.
      source_seq_len: lenght of the input sequence.
      target_seq_len: lenght of the target sequence.
      rnn_size: number of units in the rnn.
      num_layers: number of rnns to stack.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
        each timestep to compute the loss after decoding, or to feed back the
        prediction from the previous time-step.
      number_of_actions: number of classes we have.
      one_hot: whether to use one_hot encoding during train/test (sup models).
      residual_velocities: whether to use a residual connection that models velocities.
      dtype: the data type to use to store internal variables.
    """
    super(Seq2SeqModel, self).__init__()

    self.HUMAN_SIZE = 54
    self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE

    print( "One hot is ", one_hot )
    print( "Input size is %d" % self.input_size )

    # Summary writers for train and test runs

    self.source_seq_len = source_seq_len
    self.target_seq_len = target_seq_len
    self.rnn_size = rnn_size
    self.batch_size = batch_size
    self.dropout = dropout

    # === Create the RNN that will keep the state ===
    print('rnn_size = {0}'.format( rnn_size ))
    self.cell = torch.nn.GRUCell(self.input_size, self.rnn_size)
#    self.cell2 = torch.nn.GRUCell(self.rnn_size, self.rnn_size)
    self.fc1 = nn.Linear(self.rnn_size, self.input_size)


  def forward(self, encoder_inputs, decoder_inputs):
    def loop_function(prev, i):
        return prev

    batchsize = encoder_inputs.size()[0]
    encoder_inputs = torch.transpose(encoder_inputs, 0, 1)
    decoder_inputs = torch.transpose(decoder_inputs, 0, 1)

    state = torch.zeros(batchsize, self.rnn_size).\
        to(encoder_inputs.get_device())
#    state2 = torch.zeros(batchsize, self.rnn_size)
#    if use_cuda:
#        state = state.cuda()
#     #   state2 = state2.cuda()
    for i in range(self.source_seq_len-1):
        state = self.cell(encoder_inputs[i], state)
#        state2 = self.cell2(state, state2)
        state = F.dropout(state, self.dropout, training=self.training)
#        if use_cuda:
#            state = state.cuda()
##            state2 = state2.cuda()

    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      # loop function is trained as in auto regressive
      if loop_function is not None and prev is not None:
          inp = loop_function(prev, i)

      inp = inp.detach()

      state = self.cell(inp, state)
#      state2 = self.cell2(state, state2)
#      output = inp + self.fc1(state2)
#      state = F.dropout(state, self.dropout, training=self.training)
      output = inp + self.fc1(F.dropout(state, self.dropout, training=self.training))

      outputs.append(output.view([1, batchsize, self.input_size]))
      if loop_function is not None:
        prev = output

#    return outputs, state

    outputs = torch.cat(outputs, 0)
    return torch.transpose(outputs, 0, 1) 
