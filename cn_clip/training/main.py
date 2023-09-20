from math import ceil
import os
import logging
from pathlib import Path
import json
import time
from time import gmtime, strftime
import importlib.util

import torch
from torch import optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

from cn_clip.clip import load
from cn_clip.clip.model import convert_weights, convert_state_dict, resize_pos_embed, CLIP
from cn_clip.training.train import train, evaluate
from cn_clip.training.data import get_data
from cn_clip.training.params import parse_args
from cn_clip.training.logger import setup_primary_logging, setup_worker_logging
from cn_clip.training.scheduler import cosine_lr


# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def is_master(args):
    return args.rank == 0


# used to compare the pytorch version
def torch_version_str_compare_lessequal(version1, version2):
    v1 = [int(entry) for entry in version1.split("+")[0].split(".")]
    v2 = [int(entry) for entry in version2.split("+")[0].split(".")]
    assert len(v1) == 3, "Cannot parse the version of your installed pytorch! ({})".format(version1)
    assert len(v2) == 3, "Illegal version specification ({}). Should be in 1.X.Y format.".format(version2)
    return sorted([v1, v2])[0] == v1


def main():
    args = parse_args()

    # Set distributed group
    args.local_device_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_device_rank)
    args.device = torch.device("cuda", args.local_device_rank)

    dist.init_process_group(backend="nccl")
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    # Set output path
    time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    args.log_path = os.path.join(args.logs, args.name, "out_{}.log".format(time_suffix))

    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    if is_master(args):
        for dirname in [args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)    

    assert args.precision in ['amp', 'fp16', 'fp32']

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level, args.rank)

    setup_worker_logging(args.rank, log_queue, args.log_level)

    # Build the CLIP model
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)
    
    text_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)
    
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])         
        for k, v in json.load(ft).items():
            model_info[k] = v
    model_info['use_flash_attention'] = args.use_flash_attention

    model = CLIP(**model_info)
    if args.clip_weight_path is not None:
        assert os.path.exists(args.clip_weight_path), "Pretrained CLIP weight not exists!"
    if args.bert_weight_path is not None:
        assert os.path.exists(args.bert_weight_path), "Pretrained BERT weight not exists!"
    load(model, clip_path=args.clip_weight_path, bert_path=args.bert_weight_path, use_flash_attention=args.use_flash_attention)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)

    model.cuda(args.local_device_rank)
    if args.precision == "fp16":
        convert_weights(model)

    if args.grad_checkpointing:
        assert not torch_version_str_compare_lessequal(torch.__version__, "1.8.0"), \
            "Currently our grad_checkpointing is not compatible with torch version <= 1.8.0."
        model.set_grad_checkpointing()
        logging.info("Grad-checkpointing activated.")

    if args.use_flash_attention:
        assert importlib.util.find_spec("flash_attn"), "flash_attn is not installed."
        logging.info("Using FlashAttention.")

    if args.use_bn_sync:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.freeze_vision:
        for k, v in model.visual.named_parameters():
            v.requires_grad = False
        # freeze bn running mean and variance
        if args.vision_model in ['RN50']:
            for m in model.visual.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        logging.info("The visual encoder is freezed during training.")

    # To make compatible with torch version <= 1.8.0, set find_unused_parameters to True
    # In other cases, set find_unused_parameters to False
    find_unused_parameters = torch_version_str_compare_lessequal(torch.__version__, "1.8.0")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_device_rank], find_unused_parameters=find_unused_parameters)
    # Have to set this when activating grad checkpointing in Pytorch >= 2.0.0
    if args.grad_checkpointing and not torch_version_str_compare_lessequal(torch.__version__, "1.14.0"):
        model._set_static_graph()

    if args.precision == "fp16":
        convert_weights(model)

    # Initialize dataset and dataloader
    data = get_data(args, epoch_id=0, max_txt_length=args.context_length)

    # Initialize optimizer and lr scheduler
    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        num_batches = data["train"].dataloader.num_batches
        if args.max_steps is not None:
            args.max_epochs = ceil(args.max_steps * args.accum_freq / num_batches)
        else:
            assert args.max_epochs is not None and args.max_epochs > 0
            args.max_steps = (num_batches // args.accum_freq) * args.max_epochs
        total_steps = args.max_steps
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None

    # Log and save hyper-params.
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params_{}.txt".format(time_suffix))
        with open(params_file, "w", encoding="utf-8") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                f.write(f"{name}: {val}\n")

    if args.local_device_rank == 0:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
    logging.info(f"Use GPU: {args.local_device_rank} for training")

    # Note for mask_ratio
    if is_master(args) and args.mask_ratio > 0 and args.vision_model in ['RN50']:
        logging.info("Note: mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer. " + \
            "It will not function for ResNet backbone.")    

    # Optionally resume from a checkpoint
    start_epoch = 0
    steps = 0
    # Automatically restore latest checkpoint if exists
    if args.resume is None:
        latest_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
        if os.path.isfile(latest_path):
            args.resume = latest_path
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logging.info(
                f"=> begin to load checkpoint '{args.resume}'"
            )
            # Restore the model weight, map model to be loaded to specified single gpu.
            # loc = "cuda:{}".format(args.local_device_rank)
            checkpoint = torch.load(args.resume, map_location="cpu")
            sd = {k: v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
            # Resize the positional embedding by interpolation, if needed
            resize_pos_embed(sd, model, prefix="module.")
            # Adapt flash attention
            if args.use_flash_attention:
                sd = convert_state_dict(sd)
            # Load the state dict
            model.load_state_dict(sd)
            # Restore the epoch and steps info, reload the dataset and dataloader for the resume epoch
            if not args.reset_data_offset:
                start_epoch = checkpoint["epoch"]
                steps = checkpoint["step"]
                data = get_data(args, 
                                epoch_id=start_epoch, 
                                max_txt_length=args.context_length)
            # Restore the optim state
            if not args.reset_optimizer and optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
                logging.info("=> optimizer state is restored from the checkpoint")
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']} @ {steps} steps)"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False

    # determine if this worker should save logs and checkpoints.
    # only do so if it is the 0th worker.
    args.should_save = (args.logs is not None and args.logs != '' and args.logs.lower() != 'none') and is_master(args)

    # load teacher model to distllation
    if args.distllation:
        try:
            from modelscope.models import Model
        except:
            raise ImportError("modelscope is not installed. Please install it by `pip install modelscope`.")

        teacher_model_dict = {
            "damo/multi-modal_team-vit-large-patch14_multi-modal-similarity" : {"model": "image_model"},
            "damo/multi-modal_rleg-vit-large-patch14" : {"model": "encode_image"},
            "damo/multi-modal_clip-vit-huge-patch14_zh" : {"clip_model": "encode_image"},
            "damo/multi-modal_clip-vit-large-patch14_zh" : {"clip_model": "encode_image"},
        }
        assert args.teacher_model_name in teacher_model_dict, "Error: Valid teacher model name has not been built."

        try:
            teacher_model = Model.from_pretrained(args.teacher_model_name)
        except Exception as e:
            if "Unexpected key(s) in state_dict" in str(e):
                error_message = (
                    "An error occurred while loading the model: {}\n"
                    "Maybe you should update modelscope. ".format(e)
                )
                raise RuntimeError(error_message)

        for k, v in teacher_model.state_dict().items():
            v.requires_grad = False
        
        # mapping different extract_features function to same name
        mapping = teacher_model_dict[args.teacher_model_name]
        if "model" in mapping and hasattr(teacher_model, "model"):
            model_instance = getattr(teacher_model, "model")
            if hasattr(model_instance, mapping["model"]):
                setattr(teacher_model, "get_feature", getattr(model_instance, mapping["model"]))
        elif "clip_model" in mapping and hasattr(teacher_model, "clip_model"):
            model_instance = getattr(teacher_model, "clip_model")
            if hasattr(model_instance, mapping["clip_model"]):
                setattr(teacher_model, "get_feature", getattr(model_instance, mapping["clip_model"]))

        teacher_model.cuda(args.local_device_rank)
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_device_rank])
        logging.info(f"Teacher model loaded from {args.teacher_model_name}")
    else:
        teacher_model = None


    for epoch in range(start_epoch, args.max_epochs):
        if is_master(args) == 0:
            logging.info(f'Start epoch {epoch + 1}')
        if args.distllation:
            num_steps_this_epoch = train(model, data, epoch, optimizer, scaler, scheduler, args, steps, teacher_model)
        else:
            num_steps_this_epoch = train(model, data, epoch, optimizer, scaler, scheduler, args, steps)
        steps += num_steps_this_epoch

        if args.val_data is not None and args.valid_epoch_interval is not None and ((epoch + 1) % args.valid_epoch_interval) == 0:
            assert "val" in data, "Error: Valid dataset has not been built."
            if not args.use_flash_attention:
                evaluate(model, data, epoch, args, steps)
            else:
                # fp16 is needed in flash attention
                with torch.cuda.amp.autocast():
                    evaluate(model, data, epoch, args, steps)

        # if exists next epoch, reload the dataset and dataloader for the next epoch
        if epoch + 1 < args.max_epochs:
            data = get_data(args, epoch_id=epoch + 1, max_txt_length=args.context_length)

        # Saving checkpoints.
        if args.should_save and num_steps_this_epoch > 0:
            if (epoch + 1) == args.max_epochs or (
                args.save_epoch_frequency > 0 and ((epoch + 1) % args.save_epoch_frequency) == 0
            ):
                t1 = time.time()
                save_path = os.path.join(args.checkpoint_path, f"epoch{epoch + 1}.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "step": steps,
                        "name": args.name,
                        "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(model.state_dict()),
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, steps, time.time() - t1))
            
            # Save the latest params
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": steps,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, steps, time.time() - t1))


if __name__ == "__main__":
    main()
