# -*- coding: utf-8 -*-
'''
This script extracts image and text features for evaluation using ONNX model. (with CPU or single-GPU)
'''
import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
import onnxruntime
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
        "--onnx-image-model",
        default=None,
        type=str,
        help="Path to ONNX image model.",
    )
    parser.add_argument(
        "--onnx-text-model",
        default=None,
        type=str,
        help="Path to ONNX text model.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Use CPU or GPU ONNX runtime.",
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

    if args.device == "cpu":
        provider = "CPUExecutionProvider" 
    else:
        provider = "CUDAExecutionProvider"

    img_sess_options = onnxruntime.SessionOptions()
    img_run_options = onnxruntime.RunOptions()
    img_run_options.log_severity_level = 2
    img_session = onnxruntime.InferenceSession(args.onnx_image_model,
                                               sess_options=img_sess_options,
                                               providers=[provider])

    txt_sess_options = onnxruntime.SessionOptions()
    txt_run_options = onnxruntime.RunOptions()
    txt_run_options.log_severity_level = 2
    txt_session = onnxruntime.InferenceSession(args.onnx_text_model,
                                               sess_options=txt_sess_options,
                                               providers=[provider])

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
                    text_features = txt_session.run(["unnorm_text_features"], {"text": texts.numpy()})[0]
                    text_features = torch.tensor(text_features).to(args.device)
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
                    image_features = img_session.run(["unnorm_image_features"], {"image": images.numpy()})[0]
                    image_features = torch.tensor(image_features).to(args.device)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    for image_id, image_feature in zip(image_ids.tolist(), image_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                        write_cnt += 1
        print('{} image features are stored in {}'.format(write_cnt, args.image_feat_output_path))

    print("Done!")