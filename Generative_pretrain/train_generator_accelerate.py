import sys
import numpy as np
import time
import torch
import utils
import glob
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

import os

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
parser = argparse.ArgumentParser("Diffusion")
parser.add_argument('--self_condition', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=24, help='batch size')
parser.add_argument('--train_num_steps', type=int, default=40000, help='num of training epochs')
parser.add_argument('--num_timesteps', type=int, default=500, help='batch size')
parser.add_argument('--size', type=int, default=256, help='batch size')
parser.add_argument('--dataset_root', type=str, default='/home/oip/data/ProstateSeg/ProstateX/images/train', help='note for this run')
parser.add_argument('--job_name', type=str, default='expriments_name', help='job_name')

args, unparsed = parser.parse_known_args()

def main():
    accelerator = Accelerator(
        split_batches=True,
        mixed_precision = 'no'
    )
    if args.job_name != '':
        args.job_name = time.strftime("%Y%m%d-%H%M%S_") + args.job_name
        if accelerator.is_main_process:
            save_path = os.path.join("./exp", args.job_name)
            utils.create_exp_dir(save_path,scripts_to_save=glob.glob('*.py'))
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
    else:
        save_path = "./exp"
        save_path = os.path.join(save_path, 'output')
        if accelerator.is_main_process:
            utils.create_exp_dir(save_path)

    if accelerator.is_main_process:
        logging.info("args = %s", args)
        logging.info("unparsed_args = %s", unparsed)

    from denoising_diffusion_pytorch import GaussianDiffusion, Trainer
    from ResUnet_without import Unet

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        self_condition=args.self_condition,

    )

    diffusion = GaussianDiffusion(
        model,
        image_size=args.size,
        timesteps=args.num_timesteps,  # number of steps
        sampling_timesteps=150,
        beta_schedule='linear'
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = Trainer(
        diffusion,
        args.dataset_root,
        accelerator,
        train_batch_size=args.batch_size,
        train_lr=8e-5,
        train_num_steps=args.train_num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        calculate_fid=False  # whether to calculate fid during training
    )

    trainer.train()
if __name__ == '__main__':
    main()
