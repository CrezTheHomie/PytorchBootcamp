import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def download_mnist():
    mnist_train_data = datasets.MNIST(root="MNIST_data", 
                                download=True, 
                                train=True, 
                                transform=ToTensor()
                                )
    
    mnist_val_data = datasets.MNIST(root="MNIST_data", 
                                download=True, 
                                train=False, 
                                transform=ToTensor()
                                )
