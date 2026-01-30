import argparse
import os
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from style_encoder.data_loader import get_dataloader
from style_encoder.discriminator import MultiPeriodDiscriminator

# from pitch_style_encoder.model import PitchTechConverter
from style_encoder.solver import train_and_validate as trainer_pitch
from style_encoder.solver_energy import train_and_validate as trainer_energy
from style_encoder.style_encoder import TechConverter
from utils.commons import utils
from utils.commons.model_utils import load_model_and_optimizer

torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")


def main():
    cmd = parse_args()
    cfg = utils.load_config(cmd.config)
    logger.info(" > config:" + cmd.config)

    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(cfg.env.port)
    print(" > n_gpus : " + str(n_gpus))
    print(" > n_workers : " + str(os.cpu_count()))
    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(n_gpus,),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config.yaml",
        help="path to model.yaml",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="../../dataset/VocalSet",
        help="path to data directory",
    )
    parser.add_argument(
        "--feature_type", type=str, default="f0", help="f0, energy is available"
    )
    parser.add_argument("--zero_shot", action="store_true", default=False)
    args = parser.parse_args()
    return args


def run(rank, n_gpus):
    cmd = parse_args()
    args = utils.load_config(cmd.config)
    model_args = utils.load_config("configs/models/tech_encoder.yaml")

    # Select encoder type
    if cmd.feature_type == "f0" or cmd.feature_type == "pitch":
        cmd.feature_type = "f0"
        if cmd.zero_shot:
            encoder_config = model_args.pitch_tech_enc
        else:
            encoder_config = model_args.zs_pitch_tech_enc
    elif cmd.feature_type == "energy" or cmd.feature_type == "volume":
        cmd.feature_type = "energy"
        if cmd.zero_shot:
            encoder_config = model_args.energy_tech_enc
        else:
            encoder_config = model_args.zs_energy_tech_enc
    else:
        print("Wrong feature type")
        exit(0)

    dist.init_process_group(
        backend="gloo" if os.name == "nt" else "nccl",
        init_method="env://",
        world_size=n_gpus,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    net_g = TechConverter(
        encoder_config,
        len(set(args.singing_techniques.pitch_tech.values())),
        zero_shot=cmd.zero_shot,
    ).to(rank)
    net_d = MultiPeriodDiscriminator(False).to(rank)

    optimizer_g = torch.optim.AdamW(
        net_g.parameters(),
        float(args.train.optimizer.lr),
        betas=args.train.optimizer.betas,
        eps=float(args.train.optimizer.eps),
    )
    optimizer_d = torch.optim.AdamW(
        net_d.parameters(),
        float(args.train.optimizer.disc_lr),
        betas=args.train.optimizer.betas,
        eps=float(args.train.optimizer.eps),
    )

    initial_global_step, net_g, optimizer_g = load_model_and_optimizer(
        args.env.expdir, net_g, optimizer_g, device=rank
    )
    initial_global_step, net_d, optimizer_d = load_model_and_optimizer(
        args.env.expdir,
        net_d,
        optimizer_d,
        name="discriminator",
        device=rank,
    )

    train_loader, train_sampler = get_dataloader(
        args, task="train", feature_type=cmd.feature_type
    )
    if rank == 0:
        valid_loader, _ = get_dataloader(
            args, task="valid", feature_type=cmd.feature_type
        )

    # Define learning rate scheduler
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_g,
        gamma=args.train.optimizer.gamma,
        last_epoch=(
            initial_global_step // len(train_loader) - 2
            if initial_global_step != 0
            else -1
        ),
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_d,
        gamma=args.train.optimizer.gamma,
        last_epoch=(
            initial_global_step // len(train_loader) - 2
            if initial_global_step != 0
            else -1
        ),
    )

    # Set device of optimizer state
    for state in optimizer_g.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(rank)
    for state in optimizer_d.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(rank)

    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    model = [net_g, net_d]
    optimizer = [optimizer_g, optimizer_d]
    scheduler = [scheduler_g, scheduler_d]

    if cmd.feature_type == "f0":
        trainer_pitch(
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader if rank == 0 else None,
            rank,
            args,
            initial_global_step,
            train_sampler,
        )
    elif cmd.feature_type == "energy":
        trainer_energy(
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader if rank == 0 else None,
            rank,
            args,
            initial_global_step,
            train_sampler,
        )


if __name__ == "__main__":
    main()
