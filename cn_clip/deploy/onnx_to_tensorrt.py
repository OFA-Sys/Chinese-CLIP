# -*- coding: utf-8 -*-
"""
This script converts Chinese-CLIP text and vision ONNX models to TensorRT format for GPU deployment.
"""

import os
import argparse

import tensorrt as trt
from tensorrt.tensorrt import Logger, Runtime
from cn_clip.clip.utils import _MODEL_INFO
from tensorrt_utils import TensorRTShape, build_engine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-arch", 
        required=True, 
        choices=["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        help="Specify the architecture (model scale) of Chinese-CLIP model to be converted."
    )    
    parser.add_argument(
        "--convert-text",
        action="store_true",
        help="Whether to convert the text encoder (text feature extractor) into TensorRT."
    )
    parser.add_argument(
        "--text-onnx-path", 
        default=None, 
        type=str,
        help="If --convert-text is True, specify the path of the input text ONNX model."
    )
    parser.add_argument(
        "--convert-vision",
        action="store_true",
        help="Whether to convert the vision encoder (vision feature extractor) into TensorRT."
    )
    parser.add_argument(
        "--vision-onnx-path", 
        default=None, 
        type=str,
        help="If --convert-vision is True, specify the path of the input vision ONNX model."
    )
    parser.add_argument('--batch-size', default=1, type=int, help='The batch size of the TensorRT model')
    parser.add_argument(
        "--context-length", type=int, default=52, help="The padded length of input text (include [CLS] & [SEP] tokens)."
    )
    parser.add_argument(
        "--save-tensorrt-path", 
        required=True,
        type=str,
        help="Path (prefix) of the output converted TensorRT Chinese-CLIP text or vision engines."
    )
    parser.add_argument('--fp16', action='store_true',
                        help='Use when convert onnx to FP16 TensorRT model.')
    parser.add_argument('--fp32', action='store_true',
                        help='Use when convert onnx to FP32 TensorRT model.')
    parser.add_argument('--fp16-banned-ops', type=str, default='',
                        help='The ops need to be banned when building FP16 TensorRT model.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    trt_logger: Logger = trt.Logger(trt.Logger.INFO)
    runtime: Runtime = trt.Runtime(trt_logger)

    trt.init_libnvinfer_plugins(trt_logger, '')

    # ONNX -> TensorRT
    batch_size = args.batch_size
    if args.convert_text:
        seq_len = args.context_length
        text_input_shape = [TensorRTShape((batch_size, seq_len),
                                        (batch_size, seq_len),
                                        (batch_size, seq_len), 'text')]
        input_text_onnx_path = args.text_onnx_path
        assert os.path.exists(input_text_onnx_path), f"Error: The specified --text-onnx-path {input_text_onnx_path} not exists!"
    
    if args.convert_vision:
        image_size = _MODEL_INFO[args.model_arch]['input_resolution']
        vision_input_shape = [TensorRTShape((batch_size, 3, image_size, image_size),
                                            (batch_size, 3, image_size, image_size),
                                            (batch_size, 3, image_size, image_size), 'image')]
        input_vision_onnx_path = args.vision_onnx_path
        assert os.path.exists(input_vision_onnx_path), f"Error: The specified --vision-onnx-path {input_vision_onnx_path} not exists!"

    if args.fp32:
        print("---------------build TensorRT engine: FP32------------------")
        if args.convert_text:
            engine = build_engine(
                runtime=runtime,
                onnx_file_path=input_text_onnx_path,
                logger=trt_logger,
                input_shapes=text_input_shape,
                workspace_size=10000 * 1024 * 1024,
                fp16=False,
                int8=False,
            )
            text_fp32_trt_path = f"{args.save_tensorrt_path}.txt.fp32.trt"
            print(f"Saving the text FP32 TensorRT model at {text_fp32_trt_path} ...")
            with open(text_fp32_trt_path, 'wb') as f:
                f.write(bytearray(engine.serialize()))
        if args.convert_vision:
            engine = build_engine(
                runtime=runtime,
                onnx_file_path=input_vision_onnx_path,
                logger=trt_logger,
                input_shapes=vision_input_shape,
                workspace_size=10000 * 1024 * 1024,
                fp16=False,
                int8=False,
            )
            vision_fp32_trt_path = f"{args.save_tensorrt_path}.img.fp32.trt"
            print(f"Saving the vision FP32 TensorRT model at {vision_fp32_trt_path} ...")
            with open(vision_fp32_trt_path, 'wb') as f:
                f.write(bytearray(engine.serialize()))
    
    if args.fp16:
        print("-----------------build TensoRT engine: FP16----------------")
        if args.convert_text:
            engine = build_engine(
                runtime=runtime,
                onnx_file_path=input_text_onnx_path,
                logger=trt_logger,
                input_shapes=text_input_shape,
                workspace_size=10000 * 1024 * 1024,
                fp16=True,
                int8=False,
            )
            text_fp16_trt_path = f"{args.save_tensorrt_path}.txt.fp16.trt"
            print(f"Saving the text FP16 TensorRT model at {text_fp16_trt_path} ...")
            with open(text_fp16_trt_path, 'wb') as f:
                f.write(bytearray(engine.serialize()))
        if args.convert_vision:
            engine = build_engine(
                runtime=runtime,
                onnx_file_path=input_vision_onnx_path,
                logger=trt_logger,
                input_shapes=vision_input_shape,
                workspace_size=10000 * 1024 * 1024,
                fp16=True,
                int8=False,
            )
            vision_fp16_trt_path = f"{args.save_tensorrt_path}.img.fp16.trt"
            print(f"Saving the vision FP16 TensorRT model at {vision_fp16_trt_path} ...")
            with open(vision_fp16_trt_path, 'wb') as f:
                f.write(bytearray(engine.serialize()))

    print("Finished ONNX to TensorRT conversion...")
    if args.convert_text:
        if args.fp32:
            print(f">>> The text FP32 TensorRT model is saved at {text_fp32_trt_path}")
        if args.fp16:
            print(f">>> The text FP16 TensorRT model is saved at {text_fp16_trt_path}")
    if args.convert_vision:
        if args.fp32:
            print(f">>> The vision FP32 TensorRT model is saved at {vision_fp32_trt_path}")
        if args.fp16:
            print(f">>> The vision FP16 TensorRT model is saved at {vision_fp16_trt_path}")