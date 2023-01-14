# -*- coding: utf-8 -*-
'''
This script extracts image and text features for evaluation using TensorRT model. (with single-GPU)
'''
import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
from cn_clip.deploy.tensorrt_utils import TensorRTModel
from cn_clip.eval.data import get_eval_img_dataset, get_eval_txt_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--extract-image-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract image features."
    )
    parser.add_argument(
        '--extract-text-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract text features."
    )
    parser.add_argument(
        '--image-data', 
        type=str, 
        default="../Multimodal_Retrieval/lmdb/test/imgs", 
        help="If --extract-image-feats is True, specify the path of the LMDB directory storing input image base64 strings."
    )
    parser.add_argument(
        '--text-data', 
        type=str, 
        default="../Multimodal_Retrieval/test_texts.jsonl", 
        help="If --extract-text-feats is True, specify the path of input text Jsonl file."
    )
    parser.add_argument(
        '--image-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-image-feats is True, specify the path of output image features."
    )    
    parser.add_argument(
        '--text-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-image-feats is True, specify the path of output text features."
    )
    parser.add_argument(
        "--img-batch-size", type=int, default=1, help="Image batch size."
    )
    parser.add_argument(
        "--text-batch-size", type=int, default=1, help="Text batch size."
    )    
    parser.add_argument(
        "--context-length", type=int, default=52, help="The maximum length of input text (include [CLS] & [SEP] tokens)."
    )
    parser.add_argument(
        "--tensorrt-image-model",
        default=None,
        type=str,
        help="Path to TensorRT image model.",
    )
    parser.add_argument(
        "--tensorrt-text-model",
        default=None,
        type=str,
        help="Path to TensorRT text model.",
    )
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
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
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )    
    args = parser.parse_args()

    return args    


if __name__ == "__main__":
    args = parse_args()

    assert args.extract_image_feats or args.extract_text_feats, "--extract-image-feats and --extract-text-feats cannot both be False!"

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    # Get data.
    if args.extract_image_feats:
        print("Preparing image inference dataset.")
        img_data = get_eval_img_dataset(args)
    if args.extract_text_feats:
        print("Preparing text inference dataset.")
        text_data = get_eval_txt_dataset(args, max_txt_length=args.context_length)

    trt_image_model = TensorRTModel(args.tensorrt_image_model)
    trt_text_model = TensorRTModel(args.tensorrt_text_model)

    # Make inference for texts
    if args.extract_text_feats:
        print('Make inference for texts...')
        if args.text_feat_output_path is None:
            args.text_feat_output_path = "{}.txt_feat.jsonl".format(args.text_data[:-6])
        write_cnt = 0
        with open(args.text_feat_output_path, "w") as fout:
            dataloader = text_data.dataloader
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    text_ids, texts = batch
                    texts = texts.cuda()
                    text_features = trt_text_model(inputs={"text": texts})['unnorm_text_features']
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    for text_id, text_feature in zip(text_ids.tolist(), text_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"text_id": text_id, "feature": text_feature})))
                        write_cnt += 1
        print('{} text features are stored in {}'.format(write_cnt, args.text_feat_output_path))

    # Make inference for images
    if args.extract_image_feats:
        print('Make inference for images...')
        if args.image_feat_output_path is None:
            # by default, we store the image features under the same directory with the text features
            args.image_feat_output_path = "{}.img_feat.jsonl".format(
                args.text_data.replace("_texts.jsonl", "_imgs"))
        write_cnt = 0
        with open(args.image_feat_output_path, "w") as fout:
            dataloader = img_data.dataloader
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    image_ids, images = batch
                    images = images.cuda()
                    image_features = trt_image_model(inputs={"image": images})['unnorm_image_features']
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    for image_id, image_feature in zip(image_ids.tolist(), image_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                        write_cnt += 1
        print('{} image features are stored in {}'.format(write_cnt, args.image_feat_output_path))

    print("Done!")