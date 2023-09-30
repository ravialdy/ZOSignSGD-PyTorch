# ZOSignSGD-PyTorch

This is the first repository that attempts to reproduce the ZoSignSGD black-box adversarial attack published in ICLR 2019 (https://openreview.net/pdf?id=BJe-DsC5Fm) that is written purely in pytorch. What makes this repo very special is because it also supports Distributed Data Parallel (DDP) for multi-GPU training.

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

## First step : Training on CIFAR-10 for ZoSignSGD Attack Reproduction

Before implementing ZoSignSGD black-box adversarial attack, we need to train some models on CIFAR10 dataset (you can also use other datasets if you want!).

### Download and Prepare the CIFAR-10 Dataset

Before you train the model, you'll need to download and prepare the CIFAR-10 dataset. You can use the provided Python script to do this automatically. 

1. Save the Python script for downloading and preparing the CIFAR-10 dataset as `download_data.py`.
2. Run the following command to download and prepare the dataset:

    ```bash
    python download_data.py
    ```

The script will download the CIFAR-10 dataset and prepare it in the `./data/CIFAR10` directory, organizing the images by their class labels.

The first thing to do after this is to proceed to train the model.

### Train the Model

To train the model on the CIFAR-10 dataset with DDP to enable multi-GPU training, run the following command:

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

---

## Second step : ZoSignSGD Black-box Adversarial Attack

This section is focused on how to use the ZoSignSGD black-box adversarial attack, implemented purely in PyTorch. This is an extension of the primary objective of training a model on the CIFAR-10 dataset. The ZoSignSGD attack is based on the paper ["SignSGD via Zeroth-Order Oracle"](https://openreview.net/pdf?id=BJe-DsC5Fm).

### Prerequisites

In addition to the prerequisites mentioned in the CIFAR-10 training section, ensure that your trained model is saved and available for the attack.

### Understanding ZoSignSGD Class

The ZoSignSGD class in the code has the following key methods:

1. `objective_func`: Computes the objective function for the attack.
2. `early_stop_crit_fct`: Decides the early stopping criteria.
3. `grad_estimate`: Estimates the gradient for the attack.
4. `batch_attack`: Performs the actual attack.

### Usage

Initialize the ZoSignSGD class by passing your trained model and attack parameters:

```python
from zo_signSGD import ZOSignSGD

# Initialize the ZoSignSGD attacker
attacker = ZOSignSGD(model, delta=0.01, T=100, mu=0.1, q=5, const=1, k=0, variant='central')
```

Perform the attack:

```python
# Assume `orig_images` and `labels` contain a batch of original images and their corresponding true labels
adv_images = attacker.batch_attack(orig_images, labels)
```

Here, `adv_images` will contain the adversarial examples generated by the attack.

### Attack Parameters

- `delta`: Step size for ZO-SGD.
- `T`: Number of iterations.
- `mu`: Perturbation factor.
- `q`: Number of queries.
- `const`: Constant factor for loss calculation.
- `k`: Threshold value for early stopping.
- `variant`: Specify which variant of ZO-SignSGD you want to use (can be 'central', 'majority', or 'distributed').

Certainly! You can extend your README.md to include instructions on how to use the new evaluation script, `evaluate.py`. Here's how you can update it:

---

## Third step : Evaluating ZoSignSGD Black-box Adversarial Attack

After successfully training your model and understanding the ZoSignSGD attack, the next step is to evaluate the attack on the CIFAR-10 dataset. The evaluation script is contained in `evaluate.py`.

### Usage

To perform the evaluation, you can use the following command:

```bash
python evaluate.py --batch_size YOUR_BATCH_SIZE --data_path ./data/CIFAR10 --model_path ./results/best_model.pth --delta YOUR_DELTA --T YOUR_T --mu YOUR_MU --q YOUR_Q --const YOUR_CONST --k YOUR_K --variant YOUR_VARIANT
```

Here's a breakdown of the ZoSignSGD specific parameters:

- `--delta`: Step size for ZO-SGD. Default is 0.01.
- `--T`: Number of iterations. Default is 100.
- `--mu`: Perturbation factor. Default is 0.1.
- `--q`: Number of queries. Default is 5.
- `--const`: Constant factor for loss calculation. Default is 1.
- `--k`: Threshold value for early stopping. Default is 0.
- `--variant`: Specify which variant of ZO-SignSGD you want to use. Default is 'central'.

All these parameters are optional and will use the default values if not specified.

### Example

Here's an example command for running the evaluation script:

```bash
python evaluate.py --batch_size 128 --data_path ./data/CIFAR10 --model_path ./results/best_model.pth --delta 0.01 --T 100 --mu 0.1 --q 5 --const 1 --k 0 --variant central
```

This will evaluate the ZoSignSGD black-box adversarial attack on the CIFAR-10 dataset using the specified parameters and the best-performing model saved in the `./results` directory.