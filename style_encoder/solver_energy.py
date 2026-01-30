import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from modules.commons import clip_grad_value_
from modules.losses import discriminator_loss, feature_loss, generator_loss
from style_encoder.utils import *
from utils.logger import StyleEncSaver as Saver


def train_and_validate(
    model,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    device,
    configs,
    initial_global_step,
    train_sampler,
):
    fp16_run = configs.train.cache.cache_fp16
    scaler = GradScaler(enabled=configs.train.cache.cache_fp16)

    # Load model
    net_g, net_d = model
    optimizer_g, optimizer_d = optimizer
    scheduler_g, scheduler_d = scheduler

    net_g.train()
    net_d.train()

    l2_loss = nn.MSELoss()

    saver = Saver(
        configs,
        initial_global_step=initial_global_step,
    )

    global_step = initial_global_step
    for epoch in range(configs.train.optimizer.epochs):
        train_sampler.set_epoch(epoch)
        for data in train_loader:
            # Load data to GPU
            for k in data.keys():
                if not k.startswith("name"):
                    data[k] = data[k].to(device)

            vol = data["vol"].unsqueeze(-1)
            low_vol = data["low_vol"].unsqueeze(-1)
            high_vol = data["high_vol"].unsqueeze(-1)
            uv = data["uv"].unsqueeze(-1)

            with autocast(device_type="cuda", enabled=fp16_run):
                # Predict high-frequency vol contour from generator
                pred_vol, pred_high_vol = net_g(low_vol, uv, data["style_id"], high_vol)

                # Define mask
                vol_mask = uv != 0.0

                # Masking source and pred vol contours
                vol = const_signal_masking_torch(vol, vol_mask)
                high_vol = const_signal_masking_torch(high_vol, vol_mask)
                pred_vol = const_signal_masking_torch(pred_vol, vol_mask)
                pred_high_vol = const_signal_masking_torch(pred_high_vol, vol_mask)

                # Compute discriminator loss
                y_d_hat_r, y_d_hat_g, _, _ = net_d(
                    high_vol.transpose(1, 2),
                    pred_high_vol.transpose(1, 2).detach(),
                )

            optimizer_d.zero_grad()
            loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            # discriminator step
            if fp16_run:
                scaler.scale(loss_disc).backward()
                grad_norm_d = clip_grad_value_(net_d.parameters(), None)
                scaler.step(optimizer_d)
                scaler.update()
            else:
                loss_disc.backward()
                grad_norm_d = clip_grad_value_(net_d.parameters(), None)
                optimizer_d.step()

            with autocast(device_type="cuda", enabled=fp16_run):
                # Compute feedback from disriminator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(
                    high_vol.transpose(1, 2),
                    pred_high_vol.transpose(1, 2),
                )

            # Compute loss
            high_lvol_l2_loss = torch.sqrt(l2_loss(pred_high_vol, high_vol))
            fm_loss = feature_loss(fmap_r, fmap_g)
            gen_loss, _ = generator_loss(y_d_hat_g)
            loss = high_lvol_l2_loss + fm_loss + gen_loss

            # Gradient step
            optimizer_g.zero_grad()

            if fp16_run:
                scaler.scale(loss).backward()
                grad_norm_g = clip_grad_value_(net_g.parameters(), None)
                scaler.step(optimizer_g)
                scaler.update()
            else:
                loss.backward()
                grad_norm_g = clip_grad_value_(net_g.parameters(), None)
                optimizer_g.step()

            # save values for logging to tensorboard
            logging_values = {
                # loss
                "loss/total": loss.item(),
                "loss/l2_high_vol": high_lvol_l2_loss.item(),
                "loss/generator": gen_loss.item(),
                "loss/discriminator": loss_disc.item(),
                "loss/feature_matching": fm_loss.item(),
                # gradient
                "grad/net_g": grad_norm_g,
                "grad/get_d": grad_norm_d,
                # learning rate
                "lr/net_g": optimizer_g.param_groups[0]["lr"],
                "lr/net_d": optimizer_d.param_groups[0]["lr"],
            }

            # Tensorboard logging
            if device == 0:
                for key in logging_values.keys():
                    saver.log_value({f"train/{key}": logging_values[key]})

                # Print info to terminal
                if global_step % configs.train.logger.interval_log == 0:
                    print(
                        "Step {} | lr : {:.6f} | time : {} | l2_high_vol : {:.5f} | batch/s : {:.2f}".format(
                            global_step,
                            optimizer_g.param_groups[0]["lr"],
                            saver.get_total_time(),
                            logging_values["loss/l2_high_vol"],
                            configs.train.logger.interval_log
                            / saver.get_interval_time(),
                        )
                    )

            # Validation
            if device == 0:
                if (
                    global_step % configs.train.logger.interval_val == 0
                    and global_step != 0
                ):
                    print("##### Validation start #####\n")
                    net_g.eval()
                    net_d.eval()
                    test_loss = 0.0
                    with torch.inference_mode():
                        for data in valid_loader:
                            # Load data from dataloader
                            for k in data.keys():
                                if not k.startswith("name"):
                                    data[k] = data[k].to(device)

                            vol = data["vol"].unsqueeze(-1)
                            low_vol = data["low_vol"].unsqueeze(-1)
                            high_vol = data["high_vol"].unsqueeze(-1)
                            uv = data["uv"].unsqueeze(-1)

                            # Predict high-frequency vol contour
                            pred_vol, pred_high_vol = net_g(
                                low_vol, uv, data["style_id"], high_vol
                            )

                            # Define mask
                            vol_mask = uv != 0.0

                            # Masking vol contours
                            vol = const_signal_masking_torch(vol, vol_mask)
                            low_vol = const_signal_masking_torch(low_vol, vol_mask)
                            high_vol = const_signal_masking_torch(high_vol, vol_mask)
                            pred_vol = const_signal_masking_torch(pred_vol, vol_mask)
                            pred_high_vol = const_signal_masking_torch(
                                pred_high_vol, vol_mask
                            )

                            # Compute validation loss
                            test_loss += torch.sqrt(l2_loss(pred_vol, vol)).item()

                            # logging tensorboard
                            saver.log_energy_contour(
                                data["name"][0],
                                data["audio_path"][0],
                                vol.view(-1),
                                low_vol.view(-1),
                                high_vol.view(-1),
                                pred_vol.view(-1),
                                pred_high_vol.view(-1),
                                configs.audio,
                                global_step,
                            )
                    saver.log_value({"valid/total": test_loss / len(valid_loader)})
                    print(f"Validation loss | RMSE : {test_loss / len(valid_loader)}")
                    print("##### Validation end #####")

                # Save generator and discriminator
                if (
                    global_step % configs.train.logger.interval_force_save == 0
                    and global_step != 0
                ):
                    saver.save_model(net_g, optimizer_g, postfix=f"{global_step}")
                    saver.save_model(
                        net_d,
                        optimizer_d,
                        name="discriminator",
                        postfix=f"{global_step}",
                    )

                    net_g.train()
                    net_d.train()
            global_step += 1
            saver.global_step_increment()
        scheduler_g.step()
        scheduler_d.step()

        # Stop training
        if global_step == configs.train.optimizer.max_training_steps:
            print("Training Done.")
            return
