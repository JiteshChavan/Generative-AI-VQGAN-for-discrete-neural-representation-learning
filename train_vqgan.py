
import math
import time
import inspect

from dataclasses import dataclass

# ------------------------------------------------------
# data loader configs
import os
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
            
            if self.current_shard + 1 >= len(self.shards):
                self.current_epoch = self.current_epoch + 1
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            
            self.tokens = load_tokens (self.shards[self.current_shard])
            self.current_position = self.process_rank * B
            # total len (tokens) = 9B
            # 0           1                                2                                3 4 5 6 7 
            # 8B to 9B    9(x)[shard2 1B to 2B instead]    10(x)[shard2 2B to 3B instead]
            # third gpu loads 2B-3B from next shard first two load remainder of the current shard
            # if remainder is not divisible by in the last shard [a:b] loads from a to c (c < b)
        return x

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


def compute_loss (vqgan_model, discriminator, perceptual_distinguisher, x, perceptual_loss_factor, rec_loss_factor, current_step):
    
    
    # VQGAN forwrad
    reconstructed_iamges, encoding_indices, vq_loss = vqgan_model (x)
    # Discriminator Forward     
    disc_real = discriminator (x)
    disc_generated = discriminator (reconstructed_iamges)
    # VGG forward
    perceptual_loss = perceptual_distinguisher (x, reconstructed_iamges) # (B, 1, 1, 1)
    perceptual_loss = perceptual_loss.squeeze ( ) #(B)

    # formality none of the optimizers have access to vgg params but this way its explicit and efficient
    #perceptual_loss = perceptual_loss

    reconstruction_loss = F.mse_loss (x, reconstructed_iamges, reduction='none') #(B, C, H, W)
    reconstruction_loss = reconstruction_loss.mean (dim=(1,2,3)) #(B)

    perceptual_recon_loss = rec_loss_factor * reconstruction_loss + perceptual_loss_factor * perceptual_loss
    perceptual_recon_loss = perceptual_recon_loss.mean()

    # compute adversarial loss for generator if it's not zeroth epoch (VQGAN paper specs)
    is_zeroth_epoch = True if current_step < steps_per_epoch else False
    if is_zeroth_epoch:
        # don't calculate
        # Wasserstein loss, approximation of BCE (discriminator(generated), ones_like(generated))
        # TODO: Internalize throughly
        # good approximation of KLD between discriminator's predictions given generated inputs and ideal distribution of generated images being classified as real.
        # basically mean negative log likelihood is approximation of F.cross_entropy (discriminator(generated), ones_like(generated)) which is approximation of 
        # KLD between (discriminator(generated), ones_like(generated)) which is proportional to
        # BCE (discriminator(genrated), ones_like(generated))
        g_loss = 0 # replace by formula
    else:
        #calculate adversarial loss for generator
        g_loss = - torch.mean (disc_generated)

        # extract raw model to access the functions
        if ddp:
            vqgan_model = vqgan_model.module
        else:
            vqgan_model = vqgan_model

        lambda_factor = vqgan_model.compute_lambda (perceptual_recon_loss, g_loss)
        g_loss = lambda_factor * g_loss
        #if master_process:
           # print (f"\nGanLoss{g_loss:.6f} activated!\n")
        
    # VQ GAN LOSS
    # Net Generator Loss
    vq_gan_loss = perceptual_recon_loss + vq_loss + g_loss
                    
    # Discriminator Loss : Hinge Loss. Push Logits for the catgeorical distribution that come out of discriminator 
    # to be more than +1 for real images
    # At the same time Push logits for categorical distribution that comes out of discriminator to be less than -1 for fake images

    # drive logits corresponding to real images from discriminator above zero
    # don't need to explicitly squeeze out the spurious depth dimension (1 channels) from discriminator output, mean will still be evaluated correctly
    d_loss_real= torch.mean(F.relu (1.0 - disc_real))
    # drive logits corresponding to fake(generate) images from discriminator below -1
    d_loss_fake = torch.mean(F.relu (1.0 + disc_generated))

    d_loss = d_loss_factor * 0.5 * (d_loss_real + d_loss_fake)
    return vq_gan_loss, d_loss


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

# How many images to forward across all the GPUs (ddp_worldsize x) before updating weights?
# Single GPU forwards B images in one forward pass
# ensemble of GPUs forwards ddp_worldsize * B images in one forward pass
# desirable to have total batch size (total images forwarded per gradient step) to be integral multiple of ddp_world_size * B

# after n ddp forward passes we do one update i.e. total batch size is 4 * ddp world size * batchSize
# basically grad accum steps
# batch size (images in a batch)
B = 4
# total batch size as number of tokens processed per backward update

# with current setup our total batch size is 4 * 8 * 32 = 1024 images processed per gradient step
total_batch_size = 64


assert total_batch_size % (ddp_world_size * B) == 0, f"make sure total batch size is divisible by (B*T * ddp_world_size)"
grad_accum_steps = total_batch_size // (ddp_world_size * B) # each process will do B*T and theres ddp_world_size processes


if master_process: # then guard this
    print (f"total desired batch size: {total_batch_size}")
    print (f"=> calculated gradient accumulation steps: {grad_accum_steps}")


src_shards = "./dummy_tests/shards"

train_loader = DataloaderLite (B=B, num_processes=ddp_world_size, process_rank=ddp_rank, split='train', data_root=src_shards)
val_loader = DataloaderLite (B=B, num_processes=ddp_world_size, process_rank=ddp_rank, split='val', data_root=src_shards)

# find number of images in training set, to define number of epochs as a function of batch size B (n_train_iamges/B)
n_train_images = len(train_loader.shards) * len(train_loader.tokens)
print (f">Number of training images: {n_train_images}")

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

vqgan_model = VQGan (EncoderConfig, QuantizerConfig, DecoderConfig)
discriminator = Discriminator (DiscriminatorConfig)
# LPIPS/VGG
# MSE in latent space
perceptual_distinguisher = LPIPS ().eval()

vqgan_model.to(device)
discriminator.to(device)
perceptual_distinguisher.to(device)

if ddp:
    # forward is unchanged, backward is mostly unchanged except there is overlap between computation and communication of gradients
    # while the backward pass is still going on, to average the gradients from all processes
    # we're tacking on this average as we will see in a bit
    vqgan_model = DDP (vqgan_model, device_ids=[ddp_local_rank])
    discriminator = DDP (discriminator, device_ids=[ddp_local_rank])
raw_vqgan_model = vqgan_model.module if ddp else vqgan_model
raw_discriminator = discriminator.module if ddp else discriminator



from data_utils import DataUtils, Data_Utils_Config
shard_util = DataUtils (Data_Utils_Config)

# TODO: cross check against VQ GAN paper or github repository

# TODO: change these numbers later
num_epochs = 50
#steps_per_epoch = n_train_images / B if n_train_images % B == 0 else (n_train_images // B) + 1
steps_per_checkpoint = 60
steps_per_eval = 20
steps_per_inference = 3

# TODO: try to incorporate inference from both train and val shards
inference_shard_path = "./dummy_tests/shards/shard_train_0001.npy"
inference_results_path = "./dummy_tests/results"

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
min_lr = 0.1 * max_lr
steps_per_epoch = n_train_images // total_batch_size
max_steps = num_epochs * steps_per_epoch
#warmup_steps = int (0.009 * max_steps) # to match andrej's liner warmup schedule 
def get_lr (it, model):

    assert model in {"vqgan", "discriminator"}
    if model == "vqgan":
        warmup_steps = int (0.009 * max_steps)
    else:
        warmup_steps = 0

    #if model == "discriminator":
    #    assert warmup_steps_for_model == 0

    # formality, this condition wont ever be triggered because our training stops at max steps, decay till decay time and then 10% of max LR can be specified by writing 
    # the cosine coeff as a function of decay time, not of max_steps
    
    # 1) linear warmup for warmup_iters steps


    if it < warmup_steps:
        # linear warmup
        return max_lr * (it+1)/warmup_steps
        
    # 2) if it > lr_decay_iters, return min_learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + (max_lr-min_lr) * cosine_coeff

# --------------------------------------------------------------------------------------------------------------

# TODO: merge all hyper parameters in a dataclass
rec_loss_factor = 1.0
perceptual_loss_factor = 1.0
d_loss_factor = 1.0

# Optimize!!!
# First try to crush and overfit a batch
# TODO : change weight decay regularization strength later

# vqgan_optimizer has access to vqgan parameters only
vqgan_optimizer = raw_vqgan_model.configure_optimizers (weight_decay=0.1, learning_rate=max_lr, device=device)
# discriminator_optimizer has access to discriminator parameters only
disc_optimizer = raw_discriminator.configure_optimizers (weight_decay=0.1, learning_rate=max_lr, device=device)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
if master_process:
    os.makedirs (log_dir, exist_ok=True)
    log_file = os.path.join (log_dir, f"log.txt")
    with open (log_file, "w") as f: # open for writing to clear the file
        pass
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
    resume_from_checkpoint = "nobody's business"
    print (f"Resuming from checkpoint : {resume_from_checkpoint}")
    assert os.path.exists (resume_from_checkpoint), f"no checkpoint file:{resume_from_checkpoint} found"

    checkpoint = torch.load (resume_from_checkpoint)
    start_step = checkpoint['step']
    # TODO: FIX EVERYTHING LIKE THIS
    vqgan_optimizer.load_state_dict(checkpoint[f"vqgan_optim_{ddp_rank}"])
    disc_optimizer.load_state_dict(checkpoint[f"disc_optim_{ddp_rank}"])

    raw_vqgan_model.load_state_dict(checkpoint['vqgan_model'])
    raw_discriminator.load_state_dict(checkpoint['discriminator_model'])
    
    train_loader.current_shard = checkpoint['shard_state']
    train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
    train_loader.current_position = checkpoint['train_pos_GPU0'] + train_loader.B * ddp_rank

    val_loader.current_shard = checkpoint['val_shard_state']
    val_loader.tokens = load_tokens (val_loader.shards(val_loader.current_shard))
    val_loader.current_position = checkpoint['val_pos_GPU0'] + val_loader.B * ddp_rank

    print (f"\n\n\nLOADED TRAIN SHARD {train_loader.current_shard}\n\n\n")
    print (f"\n\n\nLOADED TRAIN TOKENS FROM {train_loader.shards[train_loader.current_shard]}")
    print (f"\n LOADED CURRENT TRAIN POSITION {train_loader.current_position}")

    if ddp:
        vqgan_model = DDP (raw_vqgan_model)
        discriminator = DDP (raw_discriminator)
    # hacky but it works since all pre inits are explicitly written before loading checkpoints
    raw_vqgan_model = vqgan_model.module if ddp else raw_vqgan_model
    raw_discriminator = discriminator.module if ddp else raw_discriminator
    if master_process:
        print(f"Checkpoint loaded, resuming from step {start_step}")

# TODO: Turn ON if everything is working after first dry run
use_compile = False
if use_compile:
    vqgan_model = torch.compile(vqgan_model)
    discriminator = torch.compile (discriminator)
    perceptual_distinguisher = torch.compile(perceptual_distinguisher)
    if master_process:
        print(f"Models compiled:{use_compile}")


for step in range (start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # once in a while evaluate our validation loss 
    if (step % steps_per_eval == 0 or last_step):
        vqgan_model.eval()
        discriminator.eval ()
        val_loader.reset()
        #with torch.no_grad():
        vqgan_val_loss_accum = 0
        d_val_loss_accum = 0
        val_loss_steps = 20
        
        for _ in range (val_loss_steps):

            if step >= steps_per_epoch:
                with torch.no_grad ():
                    x = val_loader.next_batch()
                    x = x.to(device) # (B, C, H, W)
                    ze = vqgan_model.module.encoder(x)
                    vq_loss, zq, encoding_indices_useless = vqgan_model.module.quantizer(ze)

                    for i in range (len(vqgan_model.module.decoder.model) - 1):
                        zq = vqgan_model.module.decoder.model[i](zq)
                
                # lambda needs dLoss/d(last_layer_decoderweights)
                # We need gradient graph for computing lambda so we forward the last layer of decoder outside no_grad context manager
                with torch.autocast (device_type=device, dtype=torch.bfloat16):
                    decoded_image = vqgan_model.module.decoder.model[-1](zq)
                    # discriminator forward
                    disc_real = discriminator (x)
                    disc_generated = discriminator (decoded_image)
                    # VGG forward (latent space L2 loss)
                    perceptual_loss = perceptual_distinguisher(x, decoded_image) # returns (B, 1, 1, 1)
                    perceptual_loss = perceptual_loss.squeeze() # get B numbers 1 for every image so that we can add onto corresponding pixel space mse

                    # reconstruction loss
                    reconstruction_loss = F.mse_loss(x, decoded_image, reduction='none')
                    reconstruction_loss = reconstruction_loss.mean(dim=(1,2,3)) # get B numbers 1 for every image (pixel space mse)
                    perc_rec_loss = perceptual_loss_factor * perceptual_loss + rec_loss_factor * reconstruction_loss
                    perc_rec_loss = perc_rec_loss.mean()
                    # wasserstein loss
                    # adversarial loss for generator
                    g_loss = -torch.mean(disc_generated) 
                    lambda_factor = raw_vqgan_model.compute_lambda (perc_rec_loss, g_loss)
                    g_loss = lambda_factor * g_loss
                    vqgan_val_loss = perc_rec_loss + g_loss + vq_loss
            else:
                with torch.no_grad ():
                    x = val_loader.next_batch()
                    x = x.to(device) # (B, C, H, W)

                    with torch.autocast (device_type=device, dtype=torch.bfloat16):
                        #compute loss function has context of global variables like steps_per_epoch, num_epochs etc
                        vqgan_val_loss, d_val_loss = compute_loss (vqgan_model=vqgan_model, discriminator=discriminator, perceptual_distinguisher=perceptual_distinguisher, x=x, perceptual_loss_factor=perceptual_loss_factor,
                                                           rec_loss_factor=rec_loss_factor, current_step=step)
                   
            # we are not interested in batchwise summation but in batchwise average of losses
            # and 20 val_loss 'mini" steps signify a batch
            vqgan_val_loss = vqgan_val_loss / val_loss_steps
            d_val_loss = d_val_loss / val_loss_steps
            # each of the processes have averaged val loss corresponding to 20 forwards 
            vqgan_val_loss_accum += vqgan_val_loss.detach()
            d_val_loss_accum += d_val_loss.detach()
            # these val losses across processes need to be averaged across all processes
            
        if ddp:
            dist.all_reduce(vqgan_val_loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(d_val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            # log
            print (f"VQGAN Validation loss: {vqgan_val_loss_accum.item():.4f}")
            print (f"Discriminator Validation loss: {d_val_loss_accum.item():.4f}")

            with open(log_file, "a") as f:
                f.write (f"{step} vqgan_val {vqgan_val_loss_accum.item():.4f} d_val {d_val_loss_accum.item():.4f}\n")

            # log checkpoints for masterprocess
            if step > 0 and (step % steps_per_checkpoint == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                if os.path.exists (checkpoint_path):
                    # other gpu went ahead and stored their optim state dicts, load the dict, set the keys
                    checkpoint = torch.load(checkpoint_path)
                    checkpoint['step'] = step
                    checkpoint['vqgan_model'] = raw_vqgan_model.state_dict()
                    checkpoint['discriminator_model'] = raw_discriminator.state_dict()
                    checkpoint[f"vqgan_optim_{ddp_rank}"] = vqgan_optimizer.state_dict()
                    checkpoint[f"disc_optim_{ddp_rank}"] = disc_optimizer.state_dict()

                    checkpoint['shard_state'] = train_loader.current_shard
                    checkpoint['val_shard_state'] = val_loader.current_shard
                    checkpoint['train_pos_GPU0'] = train_loader.current_position
                    checkpoint['val_pos_GPU0'] = val_loader.current_position
                else:
                    # if not create the checkpoint dict and save
                    checkpoint = {
                        'step' : step,
                        'vqgan_model' : raw_vqgan_model.state_dict(),
                        'discriminator_model' : raw_discriminator.state_dict(),
                        f"vqgan_optim_{ddp_rank}" : vqgan_optimizer.state_dict(),
                        f"disc_optim_{ddp_rank}" : disc_optimizer.state_dict(),
                        'shard_state' : train_loader.current_shard,
                        'val_shard_state' : val_loader.current_shard,
                        'train_pos_GPU0' : train_loader.current_position,
                        'val_pos_GPU0' : val_loader.current_position,
                    }
                print (f">__SAVING__TRAIN_SHARD_NUMBER={train_loader.current_shard}")
                print (f">__SAVING__CURRENT TRAIN POISTION FOR GPU 0={train_loader.current_position}")
                torch.save (checkpoint, checkpoint_path)
            
        # log optimizers for non master process as well
        elif not (master_process):
            checkpoint_path = os.path.join (log_dir, f"model_{step:05d}.pt")
            # if process 0 went ahead and saved its contents, load the dict, update with the processes' own optim state dicts and save back
            if os.path.exists (checkpoint_path):
                checkpoint = torch.load (checkpoint_path)
                checkpoint[f"vqgan_optim_{ddp_rank}"] = vqgan_optimizer.state_dict()
                checkpoint[f"disc_optim_{ddp_rank}"] = disc_optimizer.state_dict()
            else:
            # else create a new checkpoint dict that stores optim states of the particular GPUs and save it 
                checkpoint = {
                    f"vqgan_optim_{ddp_rank}" : vqgan_optimizer.state_dict(),
                    f"disc_optim_{ddp_rank}" : disc_optimizer.state_dict()
                }
            torch.save (checkpoint, checkpoint_path)
    
    # once in a while transform images in latent space and reconstruct from the latent representation
    # inference
    if (step > 0 and (step % steps_per_inference == 0 or last_step)):
        inference_steps = 4
        vqgan_model.eval ()
        discriminator.eval ()
        num_reconstructions = B
        inference_tokens = load_tokens (inference_shard_path)
        for i in range (inference_steps):
            # 0 : B 0:4
            # B : 2B 4:8
            # 2B : 3B 8:12
            current_buffer = inference_tokens[i*B : i*B + B]
            current_buffer = current_buffer.to(device)

            current_step_results_path = f"{inference_results_path}/{step}"
            os.makedirs(current_step_results_path, exist_ok=True)
            shard_util.tensor_to_image (current_buffer, current_step_results_path, "clone")

            with torch.no_grad():
                with torch.autocast (device_type=device, dtype=torch.bfloat16):
                    reconstructed_images, encoding_indices, vqgan_loss = vqgan_model (current_buffer)
            
                if torch.min(reconstructed_images) < -1 or torch.max(reconstructed_images) > 1:
                    reconstructed_images = torch.tanh (reconstructed_images)
            
                shard_util.tensor_to_image (reconstructed_images, current_step_results_path, "neural")
    
    # Do one step of training optimization
    vqgan_model.train()
    discriminator.train()

    vqgan_optimizer.zero_grad()
    vqgan_loss_accum = 0.0
    d_loss_accum = 0.0
    for micro_step in range (grad_accum_steps):
        x = train_loader.next_batch ()
        # ship active image batch to GPU during training to be memory efficient
        x = x.to(device)
        # bfloat16 sorcery
        #with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # generator net loss, discriminator loss
        with torch.autocast (device_type=device, dtype=torch.bfloat16):
            vqgan_loss, d_loss = compute_loss (vqgan_model=vqgan_model, discriminator=discriminator, perceptual_distinguisher=perceptual_distinguisher, x=x, perceptual_loss_factor=perceptual_loss_factor,
                                               rec_loss_factor=rec_loss_factor, current_step=step)
        # we have to scale down the losses to account for gradient accumulation
        # since consecutive loss.backward() calls deposit/add gradients
        # addition of gradients corresponds to SUM in objective
        # we do not want summation of losses corresponding to mini batches, we want average (or MEAN) instead
        # dividing losses by grad accum steps rectifies this, backward() deposits gradients corresponding to contribution of
        # one mini batch to average loss for the TOTAL_BATCH_SIZE
        vqgan_loss = vqgan_loss / grad_accum_steps
        d_loss = d_loss / grad_accum_steps

        # returns a tensor detached from computational graph, disable gradient tracking, still shares the underlying storage
        vqgan_loss_accum += vqgan_loss.detach()
        d_loss_accum += d_loss.detach()

        # only sync gradients across the processes, at the end of the large batch
        # hacky way to disable gradient sync for every single micro step

        is_last_micro_step = (micro_step == (grad_accum_steps - 1))
        if ddp:
            # very last backward will have the grad_sync flag as True
            # for now this works, but not a good practice if pytorch takes the flag away
            # averages gradients
            vqgan_model.require_backward_grad_sync = is_last_micro_step
            discriminator.require_backward_grad_sync = is_last_micro_step
        
        vqgan_loss.backward (retain_graph=True)

        disc_optimizer.zero_grad ()

        if is_last_micro_step:
            # copy from gradients from d_safe_gradients over to discriminator model parameters
            for i,p in enumerate(discriminator.parameters()):
                p.grad = d_safe_gradients[i].clone()

        d_loss.backward()
        # TODO: check
        if micro_step == 0:
            d_safe_gradients = [p.grad.clone() for p in discriminator.parameters()]
        elif not is_last_micro_step:
            for i,p in enumerate(discriminator.parameters()):
                d_safe_gradients[i] += p.grad.clone()
    
    # remember that loss.backward() always deposits gradients (grad += new_grad)
    # when we come out of the micro steps inner loop
    # every rank will suddenly, magically have the average of all the gradients on all the ranks
    if ddp:
        # calculates average of loss_accum on all the ranks, and it deposits that average on all the ranks
        # all the ranks will contain loss_accum averaged up
        dist.all_reduce (vqgan_loss_accum, op= dist.ReduceOp.AVG)
        dist.all_reduce (d_loss_accum, op=dist.ReduceOp.AVG)
    
    # global norm of parameter vector is the length of it basically.
    # clip global norm of the parameter vector, basically make sure that the "length" of the parameters vector and clip it to 1.0
    # You can get unlucky during optimization, maybe it's a bad data batch, unlucky batches yield very high loss, which could lead to very high gradients,
    # this could basically shock your model and shock your optimization.
    # so gradient norm clipping prevents model from getting too big of shocks in terms of gradient magnitudes, and it's upperbounded in this way.
    # fairly hacky solution, patch on top of deeper issues, people still do it fairly frequently.

    vqgan_norm = torch.nn.utils.clip_grad_norm_ (vqgan_model.parameters(), 1.0)
    disc_norm = torch.nn.utils.clip_grad_norm_ (discriminator.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    vqgan_lr = get_lr (step, "vqgan")
    discriminator_lr = get_lr (step, "discriminator")

    for param_group in vqgan_optimizer.param_groups:
        param_group['lr'] = vqgan_lr
    for param_group in disc_optimizer.param_groups:
        param_group['lr'] = discriminator_lr

    del d_safe_gradients

    vqgan_optimizer.step()
    disc_optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in mili seconds
    images_per_second = (train_loader.B * ddp_world_size * grad_accum_steps) / (t1 - t0)
    if master_process:
        print (f"step : {step} | vqgan_loss : {vqgan_loss_accum.item():.6f} | vqgan_norm : {vqgan_norm:.4f} | disc_loss : {d_loss_accum.item():.6f} | disc_norm : {disc_norm:.4f} | vqlr : {vqgan_lr:4e} | dlr : {discriminator_lr:4e} | "
               f"dt : {dt:2f}ms | img/sec {images_per_second:.4f}")
        with open (log_file, "a") as f:
            f.write (f"{step} vqgan_train_loss {vqgan_loss_accum.item():.6f} d_train_loss {d_loss_accum.item():6f}\n")
if ddp:
    destroy_process_group()
