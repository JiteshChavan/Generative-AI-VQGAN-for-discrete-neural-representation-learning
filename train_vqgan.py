
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
        shards = [os.path.join(data_root, s) for s in shards]
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
        x = self.tokens [self.current_position : self.current_position + B] # (B, C, H, W) B might be less than B for last shard if total images are not perfectly divible into shards, syntax works
        
        # advance current position by one block (8*B in this case)
        self.current_position += self.B * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * self.num_processes) > len (self.tokens): # len (B, C, H, W) returns B
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens (self.shards[self.current_shard])
            self.current_position = self.process_rank * B
            # total len (tokens) = 9B
            # 0           1                                2                                3 4 5 6 7 
            # 8B to 9B    9(x)[shard2 1B to 2B instead]    10(x)[shard2 2B to 3B instead]
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




# how many mini steps we should accumulate gradients before updating model weights
# jank for gradient accumulation practices for bigger datasets, not needed for smaller ones
n = 1
n_grad_accum_steps_multiplier = n * ddp_world_size

# batch size (images in a batch)
B = 64
# pixels (tokens) in image
T = 256*256

# total batch size as number of tokens processed per backward update
total_batch_size = B*T * n_grad_accum_steps_multiplier

assert total_batch_size % (ddp_world_size * B*T) == 0, f"make sure total batch size is divisible by (B*T * ddp_world_size)"
grad_accum_steps = total_batch_size // (ddp_world_size * B * T) # each process will do B*T and theres ddp_world_size processes

if master_process: # then guard this
    print (f"total desired batch size: {total_batch_size}")
    print (f"=> calculated gradient accumulation steps: {grad_accum_steps}")


train_loader = DataloaderLite (B=B, T=T, num_processes=ddp_world_size, process_rank=ddp_rank, split='train', data_root='./data/train_set')
val_loader = DataloaderLite (B=B, T=T, num_processes=ddp_world_size, process_rank=ddp_rank, split='val', data_root='./data/val_set')

# find number of images in training set, to define number of epochs as a function of batch size B (n_train_iamges/B)
n_train_images = len(train_loader.tokens)


# enable TF32 wherever possible
torch.set_float32_matmul_precision ('high')


from vqgan import VQGan
from encoder import EncoderConfig
from quantizer import QuantizerConfig
from decoder import DecoderConfig
# discriminator part of gan
from discriminator import Discriminator, DiscriminatorConfig
# for perceptual patchwise loss
from lpips import LPIPS

# Create model
# 8 exact same GPT models are created on 8 processes, because the seeds are fixed
# TODO: Refactor this jank.

model = VQGan (EncoderConfig, QuantizerConfig, DecoderConfig)
discriminator = Discriminator (DiscriminatorConfig)
# LPIPS/VGG
# MSE in latent space
perceptual_distinguisher = LPIPS.eval()

model.to(device)
discriminator.to(device)
perceptual_distinguisher.to(device)


if ddp:
    # forward is unchanged, backward is mostly unchanged except there is overlap between computation and communication of gradients
    # while the backward pass is still going on, to average the gradients from all processes
    # we're tacking on this average as we will see in a bit
    model = DDP (model, device_ids=[ddp_local_rank]) 
raw_model = model.module if ddp else model


# PyTorch has it's own cosine decay learning rate scheduler
# but it's just 5 lines of code, I know what's exactly going on
# and I don't like to use abstractions where they are kind of unscrutable and idk what they are doing


# TODO: cross check against VQ GAN paper or github repository


num_epochs = 100
steps_per_epoch = n_train_images / B if n_train_images % B == 0 else (n_train_images // B) + 1
# TODO: tweak LR later based on performance
max_lr = 6e-4
min_lr = 0.1 * max_lr
max_steps = num_epochs * steps_per_epoch
def get_lr (it):
    # formality, this condition wont ever be triggered because our training stops at max steps, decay till decay time and then 10% of max LR can be specified by writing 
    # the cosine coeff as a function of decay time, not of max_steps
    if it > max_steps:
        return min_lr
    else:
        decay_ratio = it / max_steps
        assert 0 <= decay_ratio <= 1
        cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + (max_lr-min_lr) * cosine_coeff

# TODO: merge all hyper parameters in a dataclass
rec_loss_factor = 1.0
perceptual_loss_factor = 1.0

# Optimize!!!
# First try to crush and overfit a batch
# TODO : change weight decay regularization strength later
vqgan_optimizer = raw_model.configure_optimizers (weight_decay=0.1, learning_rate=6e-4, device=device)
disc_optimizer = discriminator.configure_optimizers (weight_decay=0.1, learning_rate=6e-4, device=device)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
if master_process:
    os.makedirs (log_dir, exist_ok=True)
    log_file = os.path.join (log_dir, f"log.txt")
    with open (log_file, "w") as f: # open for writing to clear the file
        pass
    # directory to save results in
    os.makedirs("Results", exist_ok=True)


training_status = {}
# TODO: change the flag to load from previous checkpoints
fresh_run = True
resume_from_checkpoint = None if fresh_run else training_status['previous_checkpoint_file_name']

if fresh_run:
    start_step = 0
    checkpoint = {}
    if master_process:
        print("Starting a fresh run from step 0")
else:
    assert os.path.exists (resume_from_checkpoint), f"no checkpoint file:{resume_from_checkpoint} found"
    checkpoint = torch.load (resume_from_checkpoint)
    start_step = checkpoint['step']

    vqgan_optimizer.load_state_dict(checkpoint['vqgan_optim'])
    raw_model.load_state_dict(checkpoint['model'])
    
    train_loader.current_shard = checkpoint['shard_state']
    train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
    train_loader.current_position = checkpoint['current_pos_GPU0'] + train_loader.B * ddp_rank

    val_loader.current_shard = checkpoint['val_shard_state']
    val_loader.tokens = load_tokens (val_loader.shards(val_loader.current_shard))
    val_loader.current_position = checkpoint['val_pos_GPU0'] + val_loader.B * ddp_rank

    print (f"\n\n\nLOADED TRAIN SHARD {train_loader.current_shard}\n\n\n")
    print (f"\n\n\nLOADED TRAIN TOKENS FROM {train_loader.shards[train_loader.current_shard]}")
    print (f"\n LOADED CURRENT TRAIN POSITION {train_loader.current_position}")

    if ddp:
        model = DDP (raw_model)
    # hacky but it works since all pre inits are explicitly written before loading checkpoints
    raw_model = model.module if ddp else model
    if master_process:
        print(f"Checkpoint loaded, resuming from step {start_step}")

# TODO: Turn ON if everything is working after first dry run
use_compile = False
if use_compile:
    model = torch.compile(model)


for step in range (start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    # once in a while evaluate our validation loss 
    if (step % 50 == 0 or last_step):
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0
            val_loss_steps = 20
            for _ in range (val_loss_steps):
                x = val_loader.next_batch()
                x = x.to(device) # (B, C, H, W)
                with torch.autocast (device_type=device, dtype=torch.bfloat16):
                    # forward all the models involved and calculate the losses
                    # VQ forward
                    reconstructed_iamges, encoding_indices, vq_loss = model (x)
                    
                    # Discriminator Forward
                    disc_real = discriminator (x)
                    disc_generated = discriminator (reconstructed_iamges)
                    
                    # VGG forward
                    perceptual_loss = perceptual_distinguisher (x, reconstructed_iamges) # (B, 1, 1, 1)
                    perceptual_loss = perceptual_loss.squeeze() #(B)

                    reconstruction_loss = F.mse_loss (x, reconstructed_iamges, reduction='none') #(B, C, H, W)
                    reconstruction_loss = reconstruction_loss.mean (dim=(1,2,3)) #(B)

                    perceptual_recon_loss = rec_loss_factor * reconstruction_loss + perceptual_loss_factor * perceptual_loss
                    perceptual_recon_loss = perceptual_recon_loss.mean()

                    zeroth_epoch = True if step < steps_per_epoch else False
                    if zeroth_epoch == True:
                        # don't calculate
                        gan_loss = 0
                    else:
                        #calculate gan loss
                        gan_loss = 1 # replace by formula

                
                    


