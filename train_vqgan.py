
import math
import time
import inspect

from dataclasses import dataclass

# ------------------------------------------------------
# data loader configs
import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np


# TODO: CHANGE IT LATER
master_process = True


# Data loader lite only deals with shards not with images, keep that in mind
# now I know :
# how to take images from src path, scale smaller res to 256 while maintaining aspect ratio and then taking center crop
# how to take these images normalize to [-1,1] and save batches of such images in shards
# how to take a shard and load tokens into (B, C, H, W) tensor from a shard

# TODO: port the scale and center crop over to script


# ------------------------------------------------------------------------------------------------------------------------------
# Load images with pixel values normalized to [-1,1] from a specified shard as a torch tensor
# ------------------------------------------------------------------------------------------------------------------------------
def load_tokens (filename):
    npt = np.load (filename) # shape (B, H, W, C) because PIL was used to store
    ptt = torch.from_numpy (npt).permute (0, 3, 1, 2)
    return ptt

# we will make a different data utils class for reconstructions to make sure the indices of clone recons and neural recons aren't warped
class DataloaderLite:
    # to load B images for each ddp process with T= HW tokens each has 3 Channels (RGB)
    def __init__(self, B, T, num_processes, process_rank, split, data_root):
        self.B = B
        self.T = T
        self.num_processes = num_processes
        self.process_rank = process_rank
        assert split in {"train", "val"}, f"Invalid split specified at DataLoaderInstatntiation"
        assert os.path.exists (data_root)
        
        # get the shard filenames
        shards = os.listdir (data_root)
        shards = [s for s in shards if split in s]
        shards = sorted (shards)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print (f"found {len(shards)} shards for split {split}")
        # state init at shard zero
        self.reset ()
    
    def reset (self):
        self.current_shard = 0
        self.tokens = load_tokens (self.shards[self.current_shard])
        # B * T * process_rank
        # don't really need T here
        # B images go to one process
        self.current_position = self.process_rank * self.B

    def next_batch (self):
        B = self.B

        # inputs
        x = self.tokens [self.current_position : self.current_position + B] # (B, C, H, W) B might be less than B for last shard if total images are not perfectly divible into shards
        
        # advance current position by one block (8*B in this case)
        self.current_position += self.B * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * self.num_processes) > len (self.tokens): # len (B, C, H, W) returns B
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens (self.shards[self.current_shard])
            self.current_position = self.process_rank * B
            # total len (tokens) = 9B
            # 0 1 2 3 4 5 6  7
            # 8 9 10 11 12
            # k k x
            # third gpu loads 2B-3B from next shard first two load remainder of the current shard
            # if remainder is not divisible by in the last shard [a:b] loads from a to c (c < b)
        return x

# --------------------------------------------------------------------------------------------------------------
#   Optimization configurations (LR schedules, Weight decay regularization, etc)



# --------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------
# simple launch:
# python train_vqgan.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_vqgan.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# setup Distributed Data Parallel (DDP)
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE

ddp = int (os.environ.get ('RANK', -1)) != -1 # is this a ddp run or not?
if ddp:
    assert torch.cuda.is_available(), f"CUDA is required for DDP"
    init_process_group (backend='nccl')
    ddp_rank = int (os.environ['RANK'])
    ddp_local_rank = int (os.environ['LOCAL_RANK'])
    ddp_world_size = int (os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device (device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc
else:
    # non ddp vanilla run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr (torch.backends, 'mps') and torch.backends.mps.is_available ():
        device = 'mps'
    print (f"using device: {device}")

torch.manual_seed (1337)
if torch.cuda.is_available ():
    torch.cuda.manual_seed(1337)
