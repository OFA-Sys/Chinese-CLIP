# -*- coding: utf-8 -*-
"""
This script converts PyTorch implemented Chinese-CLIP (text or vision) model to ONNX format for CPU/GPU deployment.
"""

import os
import argparse
from PIL import Image
import torch
import torch.onnx
from onnx import load_model, save_model
from onnxmltools.utils import convert_float_to_float16
import cn_clip.clip as clip
from clip.utils import _MODELS, _MODEL_INFO, _download, available_models, create_model, image_transform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-arch", 
        required=True, 
        choices=["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        help="Specify the architecture (model scale) of Chinese-CLIP model to be converted."
    )
    parser.add_argument(
        "--pytorch-ckpt-path", 
        default=None, 
        type=str, 
        help="Path of the input PyTorch Chinese-CLIP checkpoint. Default to None which will automatically download the pretrained checkpoint."
    )
    parser.add_argument(
        "--download-root", 
        default=None, 
        type=str, 
        help="If --pytorch-ckpt-path is None, official pretrained ckpt will be downloaded under --download-root directory and converted. Default to ~/cache/clip/ ."
    )
    parser.add_argument(
        "--save-onnx-path", 
        required=True,
        type=str, 
        help="Path (prefix) of the output converted ONNX Chinese-CLIP text or vision model."
    )
    parser.add_argument(
        "--convert-text",
        action="store_true",
        help="Whether to convert the text encoder (text feature extractor) into ONNX."
    )
    parser.add_argument(
        "--convert-vision",
        action="store_true",
        help="Whether to convert the vision encoder (vision feature extractor) into ONNX."
    )
    parser.add_argument(
        "--context-length", type=int, default=52, help="The padded length of input text (include [CLS] & [SEP] tokens). Default to 52."
    )
    args = parser.parse_args()
    return args


def packing_small_onnx_files(onnx_path):
    # packing small files into an extra file
    save_model(load_model(onnx_path), 
            onnx_path, 
            location="{}.extra_file".format(os.path.split(onnx_path)[1]),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            size_threshold=1024,
            convert_attribute=True)
    # remove small files
    onnx_dir = os.path.split(onnx_path)[0]
    for key in checkpoint['state_dict']:
        if key.startswith('module.visual'):
            small_file_path = os.path.join(onnx_dir, key[7:])
            if os.path.exists(small_file_path):
                os.remove(small_file_path)
        if key.startswith('visual'):
            small_file_path = os.path.join(onnx_dir, key)
            if os.path.exists(small_file_path):
                os.remove(small_file_path)                    
    os.system("rm -f {}".format(os.path.join(onnx_dir, "Constant_*_attr__value")))


if __name__ == '__main__':
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    # prepare the PyTorch model weights
    if os.path.isfile(args.pytorch_ckpt_path):
        input_ckpt_path = args.pytorch_ckpt_path
    elif args.model_arch in _MODELS:
        input_ckpt_path = _download(_MODELS[args.model_arch], args.download_root or os.path.expanduser("./cache/clip"))
    else:
        raise RuntimeError(f"Model {args.model_arch} not found; available models = {available_models()}")

    with open(input_ckpt_path, 'rb') as opened_file:
        checkpoint = torch.load(opened_file, map_location="cpu")

    # prepare the PyTorch implemented model and restore weights
    model = create_model(_MODEL_INFO[args.model_arch]['struct'], checkpoint).float().eval()

    # prepare empty image and text as input placeholders for ONNX
    resolution = _MODEL_INFO[args.model_arch]['input_resolution']
    preprocess = image_transform(resolution)
    image = preprocess(Image.new('RGB', (resolution, resolution))).unsqueeze(0)
    text = clip.tokenize([""], context_length=args.context_length)

    # perform conversions, ONNX text and vision encoders will be saved into separated files
    if args.convert_text:
        # convert text FP32 ONNX model
        text_fp32_onnx_path = f"{args.save_onnx_path}.txt.fp32.onnx"
        torch.onnx.export(model,
                    (None, text),
                    text_fp32_onnx_path,
                    input_names=['text'],
                    output_names=['unnorm_text_features'],
                    export_params=True,
                    opset_version=13,
                    verbose=True)
        # convert text FP16 ONNX model based on the FP32 model
        text_fp16_onnx_path = f"{args.save_onnx_path}.txt.fp16.onnx"
        text_fp32_onnx_model = load_model(text_fp32_onnx_path)
        text_fp16_onnx_model = convert_float_to_float16(text_fp32_onnx_model, keep_io_types=True, disable_shape_infer=True)
        save_model(text_fp16_onnx_model,
                    text_fp16_onnx_path,
                    location="{}.extra_file".format(os.path.split(text_fp16_onnx_path)[1]),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    size_threshold=1024,
                    convert_attribute=True)

    if args.convert_vision:
        # convert vision FP32 ONNX model
        vision_fp32_onnx_path = f"{args.save_onnx_path}.img.fp32.onnx"
        vision_fp32_onnx_hasextra = False
        torch.onnx.export(model,
                    (image, None),
                    vision_fp32_onnx_path,
                    input_names=['image'],
                    output_names=['unnorm_image_features'],
                    export_params=True,
                    do_constant_folding=False,
                    opset_version=13,
                    verbose=True)
        # for ViT-H-14 FP32 model, make another conversion to deal with the generated small files
        if args.model_arch == "ViT-H-14":
            packing_small_onnx_files(vision_fp32_onnx_path)
            vision_fp32_onnx_hasextra = True
        # convert vision FP16 ONNX model based on the FP32 model
        vision_fp16_onnx_path = f"{args.save_onnx_path}.img.fp16.onnx"
        vision_fp32_onnx_model = load_model(vision_fp32_onnx_path)
        vision_fp16_onnx_model = convert_float_to_float16(vision_fp32_onnx_model, keep_io_types=True, disable_shape_infer=True)
        save_model(vision_fp16_onnx_model,
                    vision_fp16_onnx_path,
                    location="{}.extra_file".format(os.path.split(vision_fp16_onnx_path)[1]),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    size_threshold=1024,
                    convert_attribute=True)

    print("Finished PyTorch to ONNX conversion...")
    if args.convert_text:
        print(f">>> The text FP32 ONNX model is saved at {text_fp32_onnx_path}")
        print(f">>> The text FP16 ONNX model is saved at {text_fp16_onnx_path} with extra file {text_fp16_onnx_path}.extra_file")
    if args.convert_vision:
        print(f">>> The vision FP32 ONNX model is saved at {vision_fp32_onnx_path}" + \
            (f" with extra file {vision_fp32_onnx_path}.extra_file" if vision_fp32_onnx_hasextra else ""))
        print(f">>> The vision FP16 ONNX model is saved at {vision_fp16_onnx_path} with extra file {vision_fp16_onnx_path}.extra_file")