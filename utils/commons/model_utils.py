import glob
import logging
import os
import re
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.commons.utils import load_config

logger = logging


def traverse_dir(
    root_dir,
    extensions,
    amount=None,
    str_include=None,
    str_exclude=None,
    is_pure=False,
    is_sort=False,
    is_ext=True,
):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir) + 1 :] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list

                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue

                if not is_ext:
                    ext = pure_path.split(".")[-1]
                    pure_path = pure_path[: -(len(ext) + 1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    model = model.to(list(saved_state_dict.values())[0].dtype)
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "dec" in k or "disc" in k
            # print("load", k)
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except Exception:
            if "enc_q" not in k or "emb_g" not in k:
                print(
                    "%s is not in the checkpoint,please check your checkpoint.If you're using pretrain model,just ignore this warning."
                    % k
                )
                logger.info("%s is not in the checkpoint" % k)
                new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print("load ")
    logger.info(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def clean_checkpoints(path_to_models="logs/44k/", n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    ckpts_files = [
        f
        for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]

    def name_key(_f):
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))

    def time_key(_f):
        return os.path.getmtime(os.path.join(path_to_models, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x):
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(path_to_models, fn)
        for fn in (x_sorted("G")[:-n_ckpts_to_keep] + x_sorted("D")[:-n_ckpts_to_keep])
    ]

    def del_info(fn):
        return logger.info(f".. Free up space by deleting ckpt {fn}")

    def del_routine(x):
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def load_model(model, model_path, device):
    if not os.path.exists(model_path):
        raise FileExistsError(
            f"Model or config path is not available. Path : {model_path}"
        )

    ckpt = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    model.to(device)
    if not isinstance(model, nn.DataParallel):
        new_state_dict = OrderedDict()
        for k, v in ckpt["model"].items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(ckpt["model"], strict=True)

    model.eval()
    return model


def load_model_and_optimizer(
    expdir, model, optimizer, name="model", postfix="", device="cpu"
):
    if postfix == "":
        postfix = "_" + postfix
    path = os.path.join(expdir, name + postfix)
    path_pt = traverse_dir(expdir, ["pt"], is_ext=False)
    global_step = 0
    if len(path_pt) > 0:
        steps = [s[len(path) :] for s in path_pt]
        maxstep = max([int(s) if s.isdigit() else 0 for s in steps])
        if maxstep >= 0:
            path_pt = path + str(maxstep) + ".pt"
        else:
            path_pt = path + "best.pt"
        print(" [*] restoring model from", path_pt)
        ckpt = torch.load(path_pt, map_location=torch.device(device))
        global_step = ckpt["global_step"]

        # Multi-GPU settings for checkpoint
        if not isinstance(model, torch.nn.DataParallel):
            new_state_dict = OrderedDict()
            for k, v in ckpt["model"].items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=True)
        else:
            model.load_state_dict(ckpt["model"], strict=True)

        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
    return global_step, model, optimizer
