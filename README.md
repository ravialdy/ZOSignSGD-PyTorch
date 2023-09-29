# ZOSignSGD-PyTorch

## CIFAR-10 Training for ZoSignSGD Attack Reproduction

This repository aims to reproduce the ZoSignSGD attack and starts with training a model on the CIFAR-10 dataset. The training code is purely written in PyTorch and supports Distributed Data Parallel (DDP) for multi-GPU training.

### Prerequisites

- Python 3.x
- PyTorch 1.8+ (with CUDA if you're using GPU)
- torchvision
- tensorboard
- tqdm
- numpy

You can install the required packages using pip:

```bash
pip install torch torchvision tensorboard tqdm numpy
```

### Directory Structure

- `data/`: The folder where the CIFAR-10 dataset will be downloaded or is located.
- `results/`: The folder where the trained model checkpoints will be saved.
- `train_cifar10.py`: The Python script containing the training code.

### Training on CIFAR-10

To train the model on the CIFAR-10 dataset, run the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train_cifar10.py
```

If you want to use specific GPUs, set the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
export CUDA_VISIBLE_DEVICES=0,1   # For using the first and second GPUs
```

### TensorBoard Logging

The training and testing metrics are logged using TensorBoard. You can view them by running:

```bash
tensorboard --logdir=./results/tensorboard
```

### Checkpoints

Model checkpoints are saved in the `./results` directory. The best-performing model will be saved as `best_model.pth`, and the most recent model will be saved as `last_model.pth`.