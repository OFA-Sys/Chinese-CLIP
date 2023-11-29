# -*- coding: utf-8 -*-
"""
This script converts PyTorch implemented Chinese-CLIP (text or vision) model to CoreML format for deployment in Apple's ecosystem.
"""

import os
import argparse
from PIL import Image
import torch
from torch import nn
import coremltools as ct
import cn_clip.clip as clip
from cn_clip.clip.utils import _MODELS, _MODEL_INFO, _download, available_models, create_model, image_transform


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, image):
        return self.clip_model.encode_image(image)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, text):
        return self.clip_model.encode_text(text)


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
        help="Path of the input PyTorch Chinese-CLIP checkpoint."
    )
    parser.add_argument(
        "--download-root",
        default=None,
        type=str,
        help="If --pytorch-ckpt-path is None, official pretrained ckpt will be downloaded under --download-root directory and converted."
    )
    parser.add_argument(
        "--save-coreml-path",
        required=True,
        type=str,
        help="Path (prefix) of the output converted CoreML Chinese-CLIP text or vision model."
    )
    parser.add_argument(
        "--convert-text",
        action="store_true",
        help="Whether to convert the text encoder (text feature extractor) into CoreML."
    )
    parser.add_argument(
        "--convert-vision",
        action="store_true",
        help="Whether to convert the vision encoder (vision feature extractor) into CoreML."
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "fp32"],
        help="Specify the architecture (model scale) of Chinese-CLIP model to be converted."
    )
    parser.add_argument(
        "--context-length", type=int, default=52, help="The padded length of input text (include [CLS] & [SEP] tokens)."
    )
    args = parser.parse_args()
    return args


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
        input_ckpt_path = _download(
            _MODELS[args.model_arch], args.download_root or os.path.expanduser("./cache/clip"))
    else:
        raise RuntimeError(
            f"Model {args.model_arch} not found; available models = {available_models()}")

    with open(input_ckpt_path, 'rb') as opened_file:
        checkpoint = torch.load(opened_file, map_location="cpu")

    # prepare the PyTorch implemented model and restore weights
    model = create_model(
        _MODEL_INFO[args.model_arch]['struct'], checkpoint).float().eval()

    # prepare empty image and text as input placeholders for CoreML
    resolution = _MODEL_INFO[args.model_arch]['input_resolution']
    preprocess = image_transform(resolution)
    if args.precision == "fp16":
        precision = ct.precision.FLOAT16
    elif args.precision == "fp32":
        precision = ct.precision.FLOAT32
    image = preprocess(Image.new('RGB', (resolution, resolution))).unsqueeze(0)
    text = clip.tokenize([""], context_length=args.context_length)

    # perform conversions, CoreML text and vision encoders will be saved into separated files
    if args.convert_text:
        # Prepare the model for conversion
        text_model = TextEncoder(model)
        text_model.eval()

        # Prepare text input
        text = clip.tokenize([""], context_length=args.context_length).int()

        # Trace the model for text input
        traced_text_model = torch.jit.trace(text_model, text)

        # Convert traced model to CoreML
        text_outputs = [ct.TensorType(
            name="text_features")]
        text_coreml_model = ct.convert(
            traced_text_model,
            inputs=[ct.TensorType(name="text", shape=text.shape)],
            outputs=text_outputs,
            convert_to="mlprogram",
            compute_precision=precision,
            minimum_deployment_target=ct.target.iOS15
        )

        # Save the CoreML model
        text_coreml_model_path = f"{args.save_coreml_path}.text.mlpackage"
        text_coreml_model.save(text_coreml_model_path)
        print(
            f"Text model converted to CoreML and saved at: {text_coreml_model_path}")

    if args.convert_vision:
        # Prepare the model for conversion
        image_model = ImageEncoder(model)
        image_model.eval()

        # Prepare a dummy image input
        image_width = 336 if args.model_arch == "ViT-L-14-336" else 224
        dummy_image_input = torch.rand(1, 3, image_width, image_width)

        # Trace the model for image input
        traced_image_model = torch.jit.trace(image_model, dummy_image_input)

        # Convert traced model to CoreML
        image_outputs = [ct.TensorType(name="image_features")]
        image_coreml_model = ct.convert(
            traced_image_model,
            inputs=[ct.TensorType(
                name="image", shape=dummy_image_input.shape)],
            outputs=image_outputs,
            convert_to="mlprogram",
            compute_precision=precision,
            minimum_deployment_target=ct.target.iOS15
        )

        # Save the CoreML model
        image_coreml_model_path = f"{args.save_coreml_path}.image.mlpackage"
        image_coreml_model.save(image_coreml_model_path)
        print(
            f"Image model converted to CoreML and saved at: {image_coreml_model_path}")
