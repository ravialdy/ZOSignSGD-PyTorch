import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np

# Import the necessary modules for DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

class CIFAR10Trainer:
    """
    Initialize a trainer for CIFAR-10 dataset with DDP (Distributed Data Parallel) support.
    
    :param args: Command-line arguments containing hyperparameters and configurations.
    :type args: argparse.Namespace
    :param local_rank: The rank of the process for multi-GPU training.
    :type local_rank: int
    """
    def __init__(self, args, local_rank):
        self.args = args
        self.init_seed()
        self.init_data()
        self.init_model()
        self.init_optimizer()
        self.local_rank = local_rank
        self.writer = SummaryWriter(os.path.join(self.args.save_path, 'tensorboard'))
        
    def init_seed(self):
        """
        Initialize random seeds for reproducibility.
        """
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True

    def init_data(self):
        """
        Initialize data loaders for the CIFAR-10 dataset.
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([transforms.ToTensor()])
        train_dataset = CIFAR10(root=self.args.data_path, train=True, download=False, transform=transform_train)
        test_dataset = CIFAR10(root=self.args.data_path, train=False, download=False, transform=transform_test)
        self.train_sampler = DistributedSampler(train_dataset)
        self.test_sampler = DistributedSampler(test_dataset)
        self.trainloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, 
                                      sampler=self.train_sampler)
        self.testloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, 
                                      sampler=self.test_sampler)

    def init_model(self):
        """
        Initialize the model, set it to train on CUDA if available.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", self.local_rank)
        self.model = eval(self.args.network)(num_classes=10).to(self.device)
        self.model = DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

    def init_optimizer(self):
        """
        Initialize the optimizer and learning rate scheduler.
        """
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        self.scaler = GradScaler()

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        
        :param epoch: The current epoch number.
        :type epoch: int
        """
        self.model.train()
        total, correct, loss_total = 0, 0, 0
        for inputs, labels in tqdm(self.trainloader, desc=f"Epoch {epoch}"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            loss_total += loss.item()
        self.writer.add_scalar("train/accuracy", correct/total, epoch)
        self.writer.add_scalar("train/loss", loss_total/total, epoch)
        self.scheduler.step()

    def test_epoch(self, epoch):
        """
        Validate the model.
        
        :param epoch: The current epoch number.
        :type epoch: int
        
        :return: The test accuracy.
        :rtype: float
        """
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = correct / total
        self.writer.add_scalar("test/accuracy", accuracy, epoch)
        return accuracy

    def save_checkpoint(self, epoch, accuracy, best_acc):
        """
        Save the model checkpoint.
        
        :param epoch: The current epoch number.
        :type epoch: int
        :param accuracy: The test accuracy.
        :type accuracy: float
        :param best_acc: The best test accuracy so far.
        :type best_acc: float
        
        :return: The updated best test accuracy.
        :rtype: float
        """
        if dist.get_rank() == 0:
            state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }
            if accuracy > best_acc:
                torch.save(state, os.path.join(self.args.save_path, 'best_model.pth'))
                best_acc = accuracy
            torch.save(state, os.path.join(self.args.save_path, 'last_model.pth'))
            return best_acc

    def run(self):
        """
        The main loop to run the training and testing.
        """
        best_acc = 0
        for epoch in range(self.args.epochs):
            self.train_sampler.set_epoch(epoch)
            self.train_epoch(epoch)
            accuracy = self.test_epoch(epoch)
            best_acc = self.save_checkpoint(epoch, accuracy, best_acc)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='resnet18')
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_path', default='./data/CIFAR10')
    parser.add_argument('--save_path', default='./results')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    local_rank = dist.get_local_rank()
    torch.cuda.set_device(local_rank)

    os.makedirs(args.save_path, exist_ok=True)
    trainer = CIFAR10Trainer(args, local_rank)
    trainer.run()