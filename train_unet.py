

import math
import time
import inspect

# ------------------------------------------------------
# data loader configs
import os
import numpy as np

from dataclasses import dataclass
from data_utils import DataUtils, Data_Utils_Config

from torch.utils.tensorboard import SummaryWriter
import torch.profiler
from torch.profiler import profile, ProfilerActivity, record_function
# ------------------------------------------------------------------------------------------------------------------------------
# Load images with pixel values normalized to [-1,1] from a specified shard as a to     rch tensor
# ------------------------------------------------------------------------------------------------------------------------------
def load_tokens (filename):
    npt = np.load (filename)
    npt = npt.astype (np.float32)
    ptt = torch.tensor (npt, dtype=torch.float32)
    return ptt


# we will make a different data utils class for reconstructions to make sure the indices of clone recons and neural recons aren't warped
class DataloaderLite:
    # to load B images for each ddp process with T= HW tokens each has 3 Channels (RGB)
    def __init__(self, B, num_processes, process_rank, split, data_root):
        self.B = B
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
        self.current_epoch = 0

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

# simple launch:
# python train_vqgan.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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


B = 12
# total batch size as number of tokens processed per backward update

# with current setup our total batch size is 4 * 8 * 32 = 1024 images processed per gradient step
total_batch_size = 36


assert total_batch_size % (ddp_world_size * B) == 0, f"make sure total batch size is divisible by (B*T * ddp_world_size)"
grad_accum_steps = total_batch_size // (ddp_world_size * B) # each process will do B*T and theres ddp_world_size processes


if master_process: # then guard this
    print (f"total desired batch size: {total_batch_size}")
    print (f"=> calculated gradient accumulation steps: {grad_accum_steps}")


src_shards = "./shards"
assert os.path.exists(src_shards), f"shard path doesnt exist"

train_loader = DataloaderLite (B=B, num_processes=ddp_world_size, process_rank=ddp_rank, split='train', data_root=src_shards)
val_loader = DataloaderLite (B=B, num_processes=ddp_world_size, process_rank=ddp_rank, split='val', data_root=src_shards)

# find number of images in training set, to define number of epochs as a function of batch size B (n_train_iamges/B)
n_train_images = len(train_loader.shards) * len(train_loader.tokens)
print (f">Number of training images: {n_train_images}")

# enable TF32 wherever possible
torch.set_float32_matmul_precision ('high')


# Create model
# 8 exact same GPT models are created on 8 processes, because the seeds are fixed
# TODO: Refactor this jank.

from unet import Unet
from encoder import EncoderConfig
from decoder import DecoderConfig

unet = Unet (EncoderConfig, DecoderConfig)
unet.to(device)


if ddp:
    # forward is unchanged, backward is mostly unchanged except there is overlap between computation and communication of gradients
    # while the backward pass is still going on, to average the gradients from all processes
    # we're tacking on this average as we will see in a bit
    unet = DDP (unet, device_ids=[ddp_local_rank])
raw_unet = unet.module if ddp else unet


shard_util = DataUtils (Data_Utils_Config)

# TODO: cross check against VQ GAN paper or github repository

# TODO: change these numbers later
num_epochs = 100
#steps_per_epoch = n_train_images / B if n_train_images % B == 0 else (n_train_images // B) + 1
steps_per_checkpoint = 50
steps_per_eval = 25
steps_per_inference = 50

# TODO: try to incorporate inference from both train and val shards
inference_shard_path = "./shards/shard_train_0001.npy"
inference_results_path = "./results"

os.makedirs(inference_results_path, exist_ok=True)
assert os.path.exists(inference_shard_path), "\nNo inference shard found!\n"
assert os.path.exists (inference_results_path), "\nInvalid path specified for storing result images!\n"

assert steps_per_checkpoint % steps_per_eval == 0, f"we only checkpoint after running eval. stepsPerCheckPoint: {steps_per_checkpoint} must be divisible by stepsPerEval : {steps_per_eval}"

# TODO: tweak LR later based on performance

# PyTorch has it's own cosine decay learning rate scheduler
# but it's just 5 lines of code, I know what's exactly going on
# and I don't like to use abstractions where they are kind of unscrutable and idk what they are doing

# --------------------------------------------------------------------------------------------------------------
#   Optimization configurations (LR schedules)
# --------------------------------------------------------------------------------------------------------------

max_lr = 6e-4
min_lr = 0.7 * max_lr

steps_per_epoch = n_train_images // total_batch_size
max_steps = num_epochs * steps_per_epoch

warmup_steps = 112 # 0.9 % of max steps (12500 for current numbers)

def get_lr (it):

    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        # linear warmup
        return max_lr * (it+1)/warmup_steps
    
    # formality, this condition wont ever be triggered because our training stops at max steps, decay till decay time and then 10% of max LR can be specified by writing 
    # the cosine coeff as a function of decay time, not of max_steps
    # 2) if it > lr_decay_iters, return min_learning rate
    if it > max_steps:
        return min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + (max_lr-min_lr) * cosine_coeff

#Optimize!!!
# First try to crush and overfit a batch
# TODO : change weight decay regularization strength later

# vqgan_optimizer has access to vqgan parameters only
unet_optimizer = raw_unet.configure_optimizers (weight_decay=0.1, learning_rate=max_lr, device=device)

# create the log directory we will write checkpoints to and log to
log_dir = "logs/" 
writer = SummaryWriter(log_dir=log_dir)

if master_process:
    #os.makedirs (log_dir, exist_ok=True)
    #log_file = os.path.join (log_dir, f"log.txt")
    #with open (log_file, "w") as f: # open for writing to clear the file
    #    pass
    # directory to save results in
    os.makedirs("Results", exist_ok=True)

# TODO: change the flag to load from previous checkpoints
fresh_run = True

if fresh_run:
    start_step = 0
    checkpoint = {}
    if master_process:
        print("Starting a fresh run from step 0")
    
else:
    # TODO: change this to manually specify training checkpoint
    resume_from_checkpoint = "none of your business"
    if master_process:
        print (f"Resuming from checkpoint : {resume_from_checkpoint}")
    assert os.path.exists (resume_from_checkpoint), f"no checkpoint file:{resume_from_checkpoint} found"

    checkpoint = torch.load (resume_from_checkpoint)
    start_step = checkpoint['step']
    # TODO: FIX EVERYTHING LIKE THIS
    unet_optimizer.load_state_dict(checkpoint[f"unet_optim"])

    raw_unet.load_state_dict(checkpoint['unet_model'])
    
    train_loader.current_shard = checkpoint['shard_state']
    train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
    train_loader.current_position = checkpoint['train_pos_GPU0'] + train_loader.B * ddp_rank

    val_loader.current_shard = checkpoint['val_shard_state']
    val_loader.tokens = load_tokens (val_loader.shards[val_loader.current_shard])
    val_loader.current_position = checkpoint['val_pos_GPU0'] + val_loader.B * ddp_rank

    if master_process:
        print (f"\n\n\nLOADED TRAIN SHARD {train_loader.current_shard}\n\n\n")
        print (f"\n\n\nLOADED TRAIN TOKENS FROM {train_loader.shards[train_loader.current_shard]}")
        print (f"\n LOADED CURRENT TRAIN POSITION {train_loader.current_position}")

    if ddp:
        unet = DDP (raw_unet)
    # hacky but it works since all pre inits are explicitly written before loading checkpoints
    raw_unet = unet.module if ddp else raw_unet
    if master_process:
        print(f"Checkpoint loaded, resuming from step {start_step}")

# TODO: Turn ON if everything is working after first dry run
use_compile = False
if use_compile:
    unet = torch.compile(unet)
    if master_process:
        print(f"Models compiled:{use_compile}")

import gc



for step in range (start_step, max_steps):
    
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # once in a while evaluate our validation loss 
    if (step % steps_per_eval == 0 or last_step):
        unet.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0
            val_loss_steps = 20

            for _ in range (val_loss_steps):
                x = val_loader.next_batch()
                x = x.to(device) # (B, C, H, W)

                with torch.autocast(device_type=device, dtype=torch.bfloat16):                        
                    decoded_image = unet (x)
                    loss = F.mse_loss(x, decoded_image) # (B, C, H, W)

                    loss = loss.detach() / val_loss_steps
                    val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            # log
            print (f"Validation loss: {val_loss_accum.item():.4f}")

            writer.add_scalar("val_loss", val_loss_accum.item(), step)

            ##with open(log_file, "a") as f:
            #    f.write (f"{step} vqgan_val {val_loss_accum.item():.4f}\n")

            # log checkpoints for masterprocess
            if step > 0 and (step % steps_per_checkpoint == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                # if not create the checkpoint dict and save
                checkpoint = {
                    'step' : step,
                    'unet_model' : raw_unet.state_dict(),
                    f"unet_optim" : unet_optimizer.state_dict(),
                    'shard_state' : train_loader.current_shard,
                    'val_shard_state' : val_loader.current_shard,
                    'train_pos_GPU0' : train_loader.current_position,
                    'val_pos_GPU0' : val_loader.current_position,
                }
                print (f">__SAVING__TRAIN_SHARD_NUMBER={train_loader.current_shard}")
                print (f">__SAVING__CURRENT TRAIN POISTION FOR GPU 0={train_loader.current_position}")
                torch.save (checkpoint, checkpoint_path)
            
    # once in a while transform images in latent space and reconstruct from the latent representation
    # inference

    if (step > 0 and (step % steps_per_inference == 0 or last_step)):
        inference_steps = 5
        unet.eval ()
        num_reconstructions = 4
        inference_tokens = load_tokens (inference_shard_path)
        for i in range (inference_steps):
            # 0 : B 0:4
            # B : 2B 4:8
            # 2B : 3B 8:12
            current_buffer = inference_tokens[i*num_reconstructions : i*num_reconstructions + num_reconstructions]
            current_buffer = current_buffer.to(device)

            current_step_results_path = f"{inference_results_path}/{step}"
            os.makedirs(current_step_results_path, exist_ok=True)
            shard_util.tensor_to_image (current_buffer, current_step_results_path, "clone")

            with torch.no_grad():
                with torch.autocast (device_type=device, dtype=torch.bfloat16):
                    images = unet (current_buffer)
            
                shard_util.tensor_to_image (images, current_step_results_path, "neural")
                del images
                torch.cuda.empty_cache()

    # Do one step of training optimization
    unet.train()

    loss_accum = 0.0
    unet_optimizer.zero_grad()
    
    for micro_step in range (grad_accum_steps):
        
        x = train_loader.next_batch ()
        # ship active image batch to GPU during training to be memory efficient
        x = x.to(device)

        # bfloat16 sorcery
        #with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # generator net loss, discriminator loss

        
        with torch.autocast (device_type=device, dtype=torch.bfloat16):
            images = unet(x)
            loss = F.mse_loss (x, images)

            
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        is_last_micro_step = (micro_step == (grad_accum_steps - 1))
        if ddp:
            # very last backward will have the grad_sync flag as True
            # for now this works, but not a good practice if pytorch takes the flag away
            # averages gradients
            unet.require_backward_grad_sync = is_last_micro_step
        
        
        loss.backward()
        del images
    # remember that loss.backward() always deposits gradients (grad += new_grad)
    # when we come out of the micro steps inner loop
    # every rank will suddenly, magically have the average of all the gradients on all the ranks
    if ddp:
        # calculates average of loss_accum on all the ranks, and it deposits that average on all the ranks
        # all the ranks will contain loss_accum averaged up
        dist.all_reduce (loss_accum, op= dist.ReduceOp.AVG)
    
    # global norm of parameter vector is the length of it basically.
    # clip global norm of the parameter vector, basically make sure that the "length" of the parameters vector and clip it to 1.0
    # You can get unlucky during optimization, maybe it's a bad data batch, unlucky batches yield very high loss, which could lead to very high gradients,
    # this could basically shock your model and shock your optimization.
    # so gradient norm clipping prevents model from getting too big of shocks in terms of gradient magnitudes, and it's upperbounded in this way.
    # fairly hacky solution, patch on top of deeper issues, people still do it fairly frequently.

    norm = torch.nn.utils.clip_grad_norm_ (unet.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr (step)
    for param_group in unet_optimizer.param_groups:
        param_group['lr'] = lr

    unet_optimizer.step()
    
    
    # Optionally, force garbage collection to check for leaks


    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in mili seconds
    images_per_second = (train_loader.B * ddp_world_size * grad_accum_steps) / (t1 - t0)
    if master_process:
        print (f"step : {step} | loss : {loss_accum.item():.6f} | norm : {norm:.4f} | lr : {lr:4e} | dt : {dt:2f}ms | img/sec {images_per_second:.4f}")
        writer.add_scalar("train", loss_accum.item(), step)
        
        #with open(log_file, "a") as f:
        #        f.write (f"{step} train {loss_accum.item():.4f}\n")
writer.close()

if ddp:
    destroy_process_group()

