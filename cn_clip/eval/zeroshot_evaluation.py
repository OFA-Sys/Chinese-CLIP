# -*- coding: utf-8 -*-
'''
This script performs zero-shot evaluation on ImageNet-1K. (with single-GPU)
'''

import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm

import torch

from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.clip import tokenize
from cn_clip.training.main import convert_models_to_fp32
from cn_clip.clip.utils import image_transform
from cn_clip.eval.data import get_imagenet_dataset, _preprocess_text
from cn_clip.eval.imagenet_zeroshot_templates import imagenet_classnames, openai_imagenet_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )    
    parser.add_argument(
        "--imagenet-val",
        type=str,
        required=True,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--img-batch-size", type=int, default=64, help="Image batch size."
    )    
    parser.add_argument(
        "--context-length", 
        type=int, 
        default=32, 
        help="The maximum length of input text (include [CLS] & [SEP] tokens)."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )    
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for ImageNet dataloader."
    )        
    args = parser.parse_args()

    return args


def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [_preprocess_text(template(classname)) for template in templates] #format with class
            texts = tokenize(texts, context_length=args.context_length).to(args.gpu) #tokenize
            class_embeddings = model(None, texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.gpu)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader):
            images = images.to(args.gpu)
            target = target.to(args.gpu)

            # predict
            image_features = model(images, None)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


if __name__ == "__main__":
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    args.gpu = 0
    torch.cuda.set_device(args.gpu)

    # Initialize the model.
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

    model = CLIP(**model_info)
    convert_weights(model)    

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)
    model.cuda(args.gpu)
    if args.precision == "fp16":
        convert_weights(model)

    # Get imagenet eval data.
    print("Preparing imagenet val dataset.")
    data = {}
    data["imagenet-val"] = get_imagenet_dataset(args, image_transform(model_info['image_resolution']), "val")

    # Resume from a checkpoint.
    print("Begin to load model checkpoint from {}.".format(args.resume))
    assert os.path.exists(args.resume), "The checkpoint file {} not exists!".format(args.resume)
    # Map model to be loaded to specified single gpu.
    loc = "cuda:{}".format(args.gpu)
    checkpoint = torch.load(args.resume, map_location='cpu')
    start_epoch = checkpoint["epoch"]
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    print(
        f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']} @ {checkpoint['step']} steps)"
    )

    # Compute ensembled class embeddings
    print('Building zero-shot classifier')

    model.eval()

    classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)

    # Make inference and evaluation
    print('Using classifier')
    results = {}
    top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
    results['imagenet-zeroshot-val-top1'] = top1
    results['imagenet-zeroshot-val-top5'] = top5

    print('Result:')
    print(", ".join(["{}: {}".format(k, v) for k, v in results.items()]))
    print('Finished.')