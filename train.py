import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler

from models.data_loaders import get_dataloader
from models.model import Unit2Mel
from models.solver import train
from utils.commons import utils
from utils.commons.model_utils import load_model_and_optimizer
from vocoder import Vocoder

torch.backends.cudnn.benchmark = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    assert torch.cuda.is_available(), "CPU training is not allowed."

    cmd = parse_args()
    cfg = utils.load_config(cmd.config)
    n_gpus = torch.cuda.device_count()

    logger.info(" > config:" + cmd.config)
    logger.info(" > exp:" + cfg.env.expdir)
    logger.info(" > n_gpus : " + str(n_gpus))
    logger.info(" > n_workers : " + str(os.cpu_count()))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = cfg.env.port
    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(n_gpus,),
    )


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to the config file"
    )
    return parser.parse_args(args=args, namespace=namespace)


def run(rank, n_gpus):
    cmd = parse_args()
    args = utils.load_config(cmd.config)
    args.trainer.d_loader.num_workers = os.cpu_count()
    dist.init_process_group(
        backend="gloo" if os.name == "nt" else "nccl",
        init_method="env://",
        world_size=n_gpus,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    vocoder = Vocoder(args.model.vocoder.type, device=rank)
    model = Unit2Mel(
        args.data.encoder_out_channels,
        args.model.LUT.n_spk,
        args.model.LUT.n_t_tech,
        vocoder.dimension,
        args.model.diffusion.n_layers,
        args.model.diffusion.n_chans,
        args.model.diffusion.n_hidden,
        args.model.diffusion.timesteps,
        args.model.diffusion.k_step_max,
    ).cuda(rank)

    loader_train, train_sampler = get_dataloader(args, n_gpus, "train")
    if rank == 0:
        loader_valid, _ = get_dataloader(args, n_gpus, "valid")

    # Init optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        args.trainer.optimizer.lr,
        betas=args.trainer.optimizer.betas,
        eps=float(args.trainer.optimizer.eps),
    )

    # Load latest model and optimizer from checkpoints
    initial_global_step, model, optimizer = load_model_and_optimizer(
        args.env.expdir, model, optimizer, device=rank
    )

    logger.info(" > Loaded model to cuda " + str(rank))

    scheduler = lr_scheduler.ExponentialLR(
        optimizer,
        gamma=args.trainer.optimizer.gamma,
        last_epoch=(
            initial_global_step // len(loader_train) - 2
            if initial_global_step != 0
            else -1
        ),
    )

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(rank)

    model = DDP(model, device_ids=[rank])

    train(
        args,
        rank,
        initial_global_step,
        model,
        optimizer,
        scheduler,
        vocoder if rank == 0 else None,
        loader_train,
        loader_valid if rank == 0 else None,
        train_sampler,
    )


if __name__ == "__main__":
    main()
