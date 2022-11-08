import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

from torch.cuda.amp import autocast
import torch.distributed as dist

import logging

def is_master(args):
    return args.rank == 0

def get_loss(model, images, texts, loss_img, loss_txt, args):
    image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()
    if args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long()
    ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)

    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2

    acc = None
    if args.report_training_batch_acc:
        i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
        t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
        acc = {"i2t": i2t_acc, "t2i": t2i_acc}

    return total_loss, acc

def freeze_vision_bn(args, model):
    # freeze bn running mean and variance
    if 'RN' in args.vision_model:
        RN_visual_modules = model.module.visual.modules() if isinstance(model, nn.parallel.DistributedDataParallel) else model.visual.modules()
        for m in RN_visual_modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

def train(model, data, epoch, optimizer, scaler, scheduler, args, global_trained_steps):
    # os.environ["WDS_EPOCH"] = str(epoch)

    model.train()
    if args.freeze_vision:
        freeze_vision_bn(args, model)

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    if sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches
    data_iter = iter(dataloader)

    end = time.time()
    epoch_trained_steps = 0
    for i in range(global_trained_steps - num_batches_per_epoch * epoch, num_batches_per_epoch):
        batch = next(data_iter)
        step = num_batches_per_epoch * epoch + i
        # reach the args.max_steps, exit training:
        if step >= args.max_steps:
            logging.info("Stopping training due to step {} has reached max_steps {}".format(step, args.max_steps))
            return epoch_trained_steps
        scheduler(step)

        optimizer.zero_grad()

        images, texts, eos_indices = batch

        images = images.cuda(args.local_device_rank, non_blocking=True)
        texts = texts.cuda(args.local_device_rank, non_blocking=True)
        eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)

        data_time = time.time() - end

        m = model.module

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss, acc = get_loss(model, images, texts, loss_img, loss_txt, args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss, acc = get_loss(model, images, texts, loss_img, loss_txt, args)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        epoch_trained_steps += 1

        if is_master(args) and ((step + 1) % args.log_interval) == 0:
            num_samples = (i + 1) * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * (i + 1) / num_batches_per_epoch
            
            logging.info(
                f"Global Steps: {step + 1}/{args.max_steps} | " +
                f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                f"Loss: {total_loss.item():.6f} | " +
                (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                f"Data Time: {data_time:.3f}s | " +
                f"Batch Time: {batch_time:.3f}s | " +
                f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                f"logit_scale: {m.logit_scale.data:.3f} | " +
                f"Global Batch Size: {len(images) * args.world_size}"
            )

        if args.val_data is not None and args.valid_step_interval is not None and ((step + 1) % args.valid_step_interval) == 0:
            assert "val" in data, "Error: Valid dataset has not been built."
            evaluate(model, data, epoch, args, step + 1)
            # set model back to train mode
            model.train()
            if args.freeze_vision:
                freeze_vision_bn(args, model)

        if args.should_save and args.save_step_frequency > 0 and ((step + 1) % args.save_step_frequency) == 0:
            save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_{step + 1}.pt")
            t1 = time.time()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, step + 1, time.time() - t1))

            # Save the latest params
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, step + 1, time.time() - t1))
        
    return epoch_trained_steps


def evaluate(model, data, epoch, args, steps):

    logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, steps))

    model.eval()

    dataloader = data['val'].dataloader
    data_iter = iter(dataloader)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    cumulative_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_i2t_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_t2i_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    num_elements = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for i in range(dataloader.num_batches):
            batch = next(data_iter)
            images, texts, eos_indices = batch

            images = images.cuda(args.local_device_rank, non_blocking=True)
            texts = texts.cuda(args.local_device_rank, non_blocking=True)
            eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images)).long()
            ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)
            total_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

            cumulative_i2t_acc += ((logits_per_image.argmax(-1) == ground_truth).sum()).float()
            cumulative_t2i_acc += (logits_per_text.argmax(-1) == ground_truth).sum().float()

            if (i + 1) % 100 == 0:
                logging.info("Evaluated {}/{} batches...".format(i + 1, dataloader.num_batches))

        dist.all_reduce(cumulative_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(cumulative_i2t_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(cumulative_t2i_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_elements, op=dist.ReduceOp.SUM)
        loss = cumulative_loss / num_elements
        i2t_acc = cumulative_i2t_acc / num_elements
        t2i_acc = cumulative_t2i_acc / num_elements

        assert num_elements.item() == dataloader.num_samples # sanity check

        logging.info(
            f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
            f"Valid Loss: {loss.item():.6f} | "
            f"Image2Text Acc: {i2t_acc.item() * 100:.2f} | " 
            f"Text2Image Acc: {t2i_acc.item() * 100:.2f} | " 
            f"logit_scale: {model.module.logit_scale.data:.3f} | "
            f"Valid Batch Size: {batch_size}"
        )
