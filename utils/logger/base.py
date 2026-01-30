"""
author: wayn391@mastertones
"""

import datetime
import os
import shutil
import time

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


class BaseSaver(object):
    def __init__(self, args, initial_global_step=-1):

        self.expdir = args.env.expdir
        self.sample_rate = args.data.sampling_rate
        self.global_index = 0

        # cold start
        self.global_step = initial_global_step
        self.init_time = time.time()
        self.last_time = time.time()

        # makedirs
        os.makedirs(self.expdir, exist_ok=True)

        # path
        self.path_log_info = os.path.join(self.expdir, "log_info.txt")

        # ckpt
        os.makedirs(self.expdir, exist_ok=True)
        os.makedirs(os.path.join(self.expdir, "filelists"), exist_ok=True)

        # writer
        self.writer = SummaryWriter(os.path.join(self.expdir, "logs"))

        # save config
        path_config = os.path.join(self.expdir, "config.yaml")
        with open(path_config, "w") as out_config:
            yaml.dump(dict(args), out_config)

        path_train_flist = os.path.join(self.expdir, "filelists", args.data.flist_train)
        path_valid_flist = os.path.join(self.expdir, "filelists", args.data.flist_valid)
        shutil.copy(
            f"./filelists/{args.env.exp}/{args.data.flist_train}",
            path_train_flist,
        )
        shutil.copy(
            f"./filelists/{args.env.exp}/{args.data.flist_valid}",
            path_valid_flist,
        )

    def log_info(self, msg):
        """log method"""
        if isinstance(msg, dict):
            msg_list = []
            for k, v in msg.items():
                tmp_str = ""
                if isinstance(v, int):
                    tmp_str = "{}: {:,}".format(k, v)
                else:
                    tmp_str = "{}: {}".format(k, v)

                msg_list.append(tmp_str)
            msg_str = "\n".join(msg_list)
        else:
            msg_str = msg

        # dsplay
        print(msg_str)

        # save
        with open(self.path_log_info, "a") as fp:
            fp.write(msg_str + "\n")

    def log_value(self, dict):
        for k, v in dict.items():
            self.writer.add_scalar(k, v, self.global_step)

    def save_model(self, model, optimizer, name="model", postfix="", to_json=False):
        # path
        if postfix:
            postfix = "_" + postfix
        path_pt = os.path.join(self.expdir, name + postfix + ".pt")

        # check
        print(" [*] model checkpoint saved: {}".format(path_pt))

        # save
        if optimizer is not None:
            torch.save(
                {
                    "global_step": self.global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                path_pt,
            )
        else:
            torch.save(
                {"global_step": self.global_step, "model": model.state_dict()}, path_pt
            )

    def delete_model(self, name="model", postfix=""):
        # path
        if postfix:
            postfix = "_" + postfix
        path_pt = os.path.join(self.expdir, name + postfix + ".pt")

        # delete
        if os.path.exists(path_pt):
            os.remove(path_pt)
            print(" [*] model checkpoint deleted: {}".format(path_pt))

    def global_step_increment(self):
        self.global_step += 1

    def get_interval_time(self, update=True):
        cur_time = time.time()
        time_interval = cur_time - self.last_time
        if update:
            self.last_time = cur_time
        return time_interval

    def get_total_time(self, to_str=True):
        total_time = time.time() - self.init_time
        if to_str:
            total_time = str(datetime.timedelta(seconds=total_time))[:-5]
        return total_time
