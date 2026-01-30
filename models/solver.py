import librosa
import torch
from torch.amp import GradScaler

from modules.commons import clip_grad_value_
from utils.logger import utils
from utils.logger.saver_svc import Saver


def test(configs, rank, model, vocoder, loader_test, saver):
    print(" [*] testing...")
    model.eval()
    num_batches = len(loader_test)

    with torch.inference_mode():
        for idx, data in enumerate(loader_test):
            if idx > 20:
                break
            # unpack data
            for k in data.keys():
                if not k.startswith("name"):
                    data[k] = data[k].to(rank)

            fn = data["name"][0].split("/")[-1]
            speaker = data["name"][0].split("/")[-2]
            print("--------")
            print("{}/{} - {}".format(idx, num_batches, fn))
            print(">>", data["name"][0])

            mel = model(
                data["units"],
                data["f0"],
                data["volume"],
                data["spk_id"],
                data["tech_id"] if configs.model.LUT.n_t_tech > 1 else None,
                gt_spec=None,
                infer=True,
                infer_speedup=configs.model.infer_speedup,
                method=configs.model.infer_method,
            )

            pred_audio = vocoder.infer(mel)
            gt_vocoded = vocoder.infer(data["mel"])
            gt_audio, _ = librosa.load(data["name_ext"][0])
            if len(gt_audio.shape) > 1:
                gt_audio = librosa.to_mono(gt_audio)
            gt_audio = torch.from_numpy(gt_audio).unsqueeze(0).to(pred_audio)

            # logging to tensorboard
            saver.log_spec(f"{speaker}/{fn}", data["mel"], mel)
            saver.log_audio(
                {
                    f"{speaker}/{fn} | GT": gt_audio,
                    f"{speaker}/{fn} | Vocoded": gt_vocoded,
                    f"{speaker}/{fn} | Pred": pred_audio,
                }
            )
    return


def train(
    configs,
    rank,
    initial_global_step,
    model,
    optimizer,
    scheduler,
    vocoder,
    loader_train,
    loader_test,
    train_sampler,
):
    if rank == 0:
        # saver
        saver = Saver(configs, initial_global_step=initial_global_step)

        # model size
        params_count = utils.get_network_paras_amount({"model": model})
        saver.log_info("--- model size ---")
        saver.log_info(params_count)
        saver.log_info("======= start training =======")
        saver.log_info("epoch|batch_idx/num_batches|output_dir|batch/s|lr|time|step")

    # run
    num_batches = len(loader_train)
    model.train()
    scaler = GradScaler()
    dtype = torch.float16 if configs.trainer.cache.cache_fp16 else torch.float32

    ############## Epoch 시작 ###############
    for epoch in range(configs.trainer.optimizer.epochs):
        train_sampler.set_epoch(epoch)
        for batch_idx, data in enumerate(loader_train):
            optimizer.zero_grad()

            for k in data.keys():
                if not k.startswith("name"):
                    data[k] = data[k].to(rank)

            # forward
            if dtype == torch.float32:
                loss = model(
                    data["units"].float(),
                    data["f0"],
                    data["volume"],
                    data["spk_id"],
                    data["tech_id"] if configs.model.LUT.n_t_tech > 1 else None,
                    gt_spec=data["mel"].float(),
                )
            else:
                with torch.autocast(device_type=f"cuda:{rank}", dtype=dtype):
                    loss = model(
                        data["units"],
                        data["f0"],
                        data["volume"],
                        data["spk_id"],
                        data["tech_id"] if configs.model.LUT.n_t_tech > 1 else None,
                        gt_spec=data["mel"],
                    )
            if torch.isnan(loss):
                raise ValueError(" [x] nan loss ")
            else:
                if dtype == torch.float32:
                    loss.backward()
                    clip_grad_value_(model.parameters(), None)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    clip_grad_value_(model.parameters(), None)
                    scaler.step(optimizer)
                    scaler.update()

            # log loss
            if rank == 0:
                if saver.global_step % configs.trainer.logger.interval_log == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    saver.log_info(
                        "epoch: {} | {:2d}/{:2d} | {} | batch/s: {:.2f} | lr: {:.3} | loss: {:.3f} | time: {} | step: {}".format(
                            epoch,
                            batch_idx,
                            num_batches,
                            configs.env.expdir,
                            configs.trainer.logger.interval_log
                            / saver.get_interval_time(),
                            current_lr,
                            loss.item(),
                            saver.get_total_time(),
                            saver.global_step,
                        )
                    )
                    saver.log_value({"train/loss": loss.item()})
                    saver.log_value({"train/lr": current_lr})

                # validation
                if (
                    saver.global_step % configs.trainer.logger.interval_val == 0
                    and saver.global_step != 0
                ):
                    # Save model
                    saver.save_model(
                        model,
                        optimizer if configs.trainer.logger.save_opt else None,
                        postfix=f"{saver.global_step}",
                    )
                    last_val_step = (
                        saver.global_step - configs.trainer.logger.interval_val
                    )
                    if last_val_step % configs.trainer.logger.interval_force_save != 0:
                        saver.delete_model(postfix=f"{last_val_step}")

                    # Generate Mel-Spec to tensorboard for validation
                    test(configs, rank, model, vocoder, loader_test, saver)
                    saver.log_info(f" ==== Validation End ==== ")
                    model.train()
                saver.global_step_increment()
        scheduler.step()

        # Stop training
        if saver.global_step == configs.train.optimizer.max_training_steps:
            print("Training Done.")
            return
