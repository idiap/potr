# Pose Transformers: Human Motion Prediction with Non-Autoregressive Transformers

![alt text](imgs_docs/potr_walking.gif)

This is the repo used for human motion prediction with non-autoregressive
transformers published with our paper ![alt text]()

## Requirements

* **Pytorch**>=1.7.
* **Numpy**.
* **Tensorboard** for pytorch.

## Data

We have performed experiments with 2 different datasets

1. **H36M**
2. **NTURGB+D**

Follow the instructions to download each dataset and place it in ```data```.

## Training

To run training with H3.6M dataset and save in ```POTR_OUT`` folder
run the following:


```
python src/training/transformer_model_fn.py \
  --model_prefix=${POTR_OUT} \
  --batch_size=16 \
  --data_path=${H36M} \
  --learning_rate=0.0001 \
  --max_epochs=500 \
  --steps_per_epoch=200 \
  --loss_fn=l1 \
  --model_dim=128 \
  --num_encoder_layers=4 \
  --num_decoder_layers=4 \
  --num_heads=4 \
  --dim_ffn=2048 \
  --dropout=0.3 \
  --lr_step_size=400 \
  --learning_rate_fn=step \
  --warmup_epochs=100 \
  --pose_format=rotmat \
  --pose_embedding_type=gcn_enc \
  --dataset=h36m_v2 \
  --pre_normalization \
  --pad_decoder_inputs \
  --non_autoregressive \
  --pos_enc_alpha=10 \
  --pos_enc_beta=500 \
  --predict_activity 
```

Where ```pose_embedding_type``` controls the type of architectures of networks 
to be used for encoding and decoding skeletons (\phi and \psi in our paper). 
See ```models/PoseEncoderDecoder.py``` for the types of architectures.
Tensorboard curves and pytorch models will be saved in ```${POTR_OUT}```.



