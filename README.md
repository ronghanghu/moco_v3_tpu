# SimCLR and MoCo v3 ViT implementation

This repo implements the SimCLR and MoCo v3 algorithm on Vision Transformers (ViT) for both GPUs and TPUs, with hyperparams following [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.02057.pdf).

## Installation

Install [pytorch (and its dependencies)](https://pytorch.org/). Install [pytorch xla](https://github.com/pytorch/xla/blob/master/CONTRIBUTING.md) if running on TPUs.

Finally, install [timm](https://rwightman.github.io/pytorch-image-models/) for vision transformers: `pip3 install timm`.

Download [ImageNet-1k](https://image-net.org/) to a shared directory (e.g. to /checkpoint/ronghanghu/megavlt_paths/imagenet-1k) that can be accessed from all nodes, which should have the following structure.
```
/checkpoint/ronghanghu/megavlt_paths/imagenet-1k
|_ train
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
|_ val
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
```

## Running MoCo v3 ViT training on ImageNet-1k

Launch the training on GPUs or TPUs as follows. It trains MoCo v3 for 300 epochs by default. (Append `num_epochs=100` to the commands below to train only for 100 epochs.)

**Make sure `SAVE_DIR` is a shared directory that can be accessed from all nodes. For TPUs, one can use an NFS directory on GCP.**

On GPUs (e.g. using 64 V100 GPUs):
```
SAVE_DIR="/private/home/ronghanghu/workspace/simclr_vit_release/save_mocov3_gpu64"

srun \
  --mem=300g --nodes=8 --gres=gpu:8 --partition=learnlab,learnfair \
  --time=4300 --constraint=volta32gb --cpus-per-task=40 \
python3 run_mocov3_vit.py \
  world_size=64 \
  ckpt_dir=$SAVE_DIR \
  data_dir=/checkpoint/ronghanghu/megavlt_paths/imagenet-1k \
  batch_size=4096 lr=2.4e-3 weight_decay=0.1  # lr is already scaled by batch size
```
(append `use_pytorch_amp=True` to the command above to use automatic mixed precision)

On TPUs (e.g. using a v3-256 TPU pod):
```
SAVE_DIR="/checkpoint/ronghanghu/workspace/simclr_vit_release/save_mocov3_tpu_v3-256"

TPU_NAME=megavlt-256  # change to your TPU name
# use absolute paths with torch_xla.distributed.xla_dist
sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR  # workaround for permission issue
python3 -m torch_xla.distributed.xla_dist \
  --tpu=${TPU_NAME} --restart-tpuvm-pod \
  --env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 \
  -- \
python3 $(realpath run_mocov3_vit.py) \
  device=xla \
  ckpt_dir=$SAVE_DIR \
  data_dir=/checkpoint/ronghanghu/megavlt_paths/imagenet-1k \
  batch_size=4096 lr=2.4e-3 weight_decay=0.1  # lr is already scaled by batch size
```

## Running SimCLR ViT training on ImageNet-1k

Launch the training on GPUs or TPUs as follows. It trains SimCLR for 300 epochs by default. (Append `num_epochs=100` to the commands below to train only for 100 epochs.)

**Make sure `SAVE_DIR` is a shared directory that can be accessed from all nodes. For TPUs, one can use an NFS directory on GCP.**

On GPUs (e.g. using 64 V100 GPUs):
```
SAVE_DIR="/private/home/ronghanghu/workspace/simclr_vit_release/save_simclr_gpu64"

srun \
  --mem=300g --nodes=8 --gres=gpu:8 --partition=learnlab,learnfair \
  --time=4300 --constraint=volta32gb --cpus-per-task=40 \
python3 run_simclr_vit.py \
  world_size=64 \
  ckpt_dir=$SAVE_DIR \
  data_dir=/checkpoint/ronghanghu/megavlt_paths/imagenet-1k \
  batch_size=4096 lr=3.2e-3 weight_decay=0.1  # lr is already scaled by batch size
```
(append `use_pytorch_amp=True` to the command above to use automatic mixed precision)

On TPUs (e.g. using a v3-256 TPU pod):
```
SAVE_DIR="/checkpoint/ronghanghu/workspace/simclr_vit_release/save_simclr_tpu_v3-256"

TPU_NAME=megavlt-256  # change to your TPU name
# use absolute paths with torch_xla.distributed.xla_dist
sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR  # workaround for permission issue
python3 -m torch_xla.distributed.xla_dist \
  --tpu=${TPU_NAME} --restart-tpuvm-pod \
  --env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 \
  -- \
python3 $(realpath run_simclr_vit.py) \
  device=xla \
  ckpt_dir=$SAVE_DIR \
  data_dir=/checkpoint/ronghanghu/megavlt_paths/imagenet-1k \
  batch_size=4096 lr=3.2e-3 weight_decay=0.1  # lr is already scaled by batch size
```

## Running linear evaluation on the trained MoCo v3 or SimCLR models

Suppose the final checkpoint from the previous step is `PRETRAINED_MODEL` (e.g. `/private/home/ronghanghu/workspace/simclr_vit_release/save_simclr_gpu64/vit_b16_epoch_300.ckpt` or any checkpoint trained by SimCLR or MoCo v3 above). Let's evaluate it as follows.

* For MoCo v3 (ImageNet-1k 300 epochs by default), expected linear evaluation accuracy is around 0.765 for both GPUs and TPUs.
* For SimCLR (ImageNet-1k 300 epochs by default), expected linear evaluation accuracy is around 0.739 for both GPUs and TPUs.

**Make sure `SAVE_DIR` is a shared directory that can be accessed from all nodes. For TPUs, one can use an NFS directory on GCP.**

On GPUs (e.g. using 64 V100 GPUs):
```
PRETRAINED_MODEL=/private/home/ronghanghu/workspace/simclr_vit_release/save_simclr_gpu64/vit_b16_epoch_300.ckpt
# SAVE_DIR can be the same or a different directory from SSL training
SAVE_DIR="/private/home/ronghanghu/workspace/simclr_vit_release/save_simclr_gpu64"

srun \
  --mem=300g --nodes=8 --gres=gpu:8 --partition=learnlab,learnfair \
  --time=4300 --constraint=volta32gb --cpus-per-task=40 \
python3 $(realpath run_linear_eval_vit.py) \
  world_size=64 \
  ckpt_dir=$SAVE_DIR \
  data_dir=/checkpoint/ronghanghu/megavlt_paths/imagenet-1k \
  linear_eval.pretrained_ckpt_path=$PRETRAINED_MODEL
```

On TPUs (e.g. using a v3-256 TPU pod):
```
PRETRAINED_MODEL=/checkpoint/ronghanghu/workspace/simclr_vit_release/save_simclr_tpu_v3-256/vit_b16_epoch_300.ckpt
# SAVE_DIR can be the same or a different directory from SSL training
SAVE_DIR="/checkpoint/ronghanghu/workspace/simclr_vit_release/save_simclr_tpu_v3-256"

TPU_NAME=megavlt-256  # change to your TPU name
# use absolute paths with torch_xla.distributed.xla_dist
sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR  # workaround for permission issue
python3 -m torch_xla.distributed.xla_dist \
  --tpu=${TPU_NAME} --restart-tpuvm-pod \
  --env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 \
  -- \
python3 $(realpath run_linear_eval_vit.py) \
  device=xla \
  ckpt_dir=$SAVE_DIR \
  data_dir=/checkpoint/ronghanghu/megavlt_paths/imagenet-1k \
  linear_eval.pretrained_ckpt_path=$PRETRAINED_MODEL
```
