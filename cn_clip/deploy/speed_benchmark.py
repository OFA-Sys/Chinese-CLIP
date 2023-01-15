# -*- coding: utf-8 -*-
"""
This script runs to compare the inference speed between Pytorch (CPU/GPU), ONNX (CPU/GPU) and TensorRT models.
"""

import argparse
import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip.utils import create_model, _MODEL_INFO, image_transform
from cn_clip.training.main import convert_models_to_fp32, convert_weights
from cn_clip.deploy.benchmark_utils import track_infer_time, print_timings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-arch", 
        required=True, 
        choices=["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        help="Specify the architecture (model scale) of Chinese-CLIP model for speed comparison."
    )
    parser.add_argument('--pytorch-ckpt', type=str, default=None, 
                        help='The file path of pytorch checkpoint, if not set, the program will not run Pytorch model.')
    parser.add_argument('--pytorch-precision', choices=["fp16", "fp32"], default="fp16", 
                        help='Flag for Pytorch model float point precision, default to using FP16 for evaluation.')
    
    parser.add_argument('--onnx-image-model', type=str, default=None,
                        help='The file path of onnx image model, if not set, the program will not run image ONNX model.')
    parser.add_argument('--onnx-text-model', type=str, default=None,
                        help='The file path of onnx text model, if not set, the program will not run text ONNX model.')

    parser.add_argument('--tensorrt-image-model', type=str, default=None,
                        help='The file path of image TensorRT model, if not set, the program will not run image TensorRT model.')
    parser.add_argument('--tensorrt-text-model', type=str, default=None,
                        help='The file path of text TensorRT model, if not set, the program will not run text TensorRT model.')

    parser.add_argument('--batch-size', default=1, type=int, help='The batch-size of the inference input. Default to 1.')
    parser.add_argument('--n', default=100, type=int, help='The iteration number for inference speed test. Default to 100.')
    parser.add_argument('--warmup', default=10, type=int, help='Warmup iterations. Default to 10.')

    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="CPU or GPU speed test. Default to cuda",)
    parser.add_argument(
        "--context-length", type=int, default=52, help="The padded length of input text (include [CLS] & [SEP] tokens). Default to 52."
    )

    args = parser.parse_args()

    return args


def prepare_pytorch_model(args):
    with open(args.pytorch_ckpt, 'rb') as opened_file:
        checkpoint = torch.load(opened_file, map_location="cpu")
    pt_model = create_model(_MODEL_INFO[args.model_arch]['struct'], checkpoint)
    pt_model.eval()
    if args.pytorch_precision == "fp16":
        convert_weights(pt_model)
    else:
        convert_models_to_fp32(pt_model)
    if args.device == "cuda":
        pt_model.cuda()
    return pt_model


onnx_execution_provider_map = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


if __name__ == '__main__':
    args = parse_args()
    
    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    preprocess = image_transform(_MODEL_INFO[args.model_arch]['input_resolution'])
    image = torch.vstack([preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0)] * args.batch_size)
    text = torch.vstack([clip.tokenize(["杰尼龟"], context_length=args.context_length)] * args.batch_size)
    if args.device == "cuda":
        image = image.cuda()
        text = text.cuda()

    # test the image feature extraction
    print("Begin the image feature extraction speed test...")
    if args.pytorch_ckpt:
        print(f"Prepare the Pytorch model from {args.pytorch_ckpt}")
        pt_model = prepare_pytorch_model(args)
        for i in range(args.warmup):
            pytorch_output = pt_model(image=image, text=None)
        print("Forward the Pytorch image model...")
        time_buffer = list()
        for i in range(args.n):
            with track_infer_time(time_buffer):
                pytorch_output = pt_model(image=image, text=None)
        print_timings(name=f"Pytorch image inference speed (batch-size: {args.batch_size}):", timings=time_buffer)
        del pt_model

    if args.onnx_image_model:
        import onnxruntime
        print(f'Prepare the ONNX image model from {args.onnx_image_model}')
        sess_options = onnxruntime.SessionOptions()
        run_options = onnxruntime.RunOptions()
        run_options.log_severity_level = 2
        session = onnxruntime.InferenceSession(args.onnx_image_model,
                                                sess_options=sess_options,
                                                providers=[onnx_execution_provider_map[args.device]]
                                                )
        for i in range(args.warmup):
            onnx_output = session.run(["unnorm_image_features"], {"image": image.cpu().numpy()})
        print("Forward the ONNX image model...")
        time_buffer = list()
        for i in range(args.n):
            with track_infer_time(time_buffer):
                onnx_output = session.run(["unnorm_image_features"], {"image": image.cpu().numpy()})
        print_timings(name=f"ONNX image inference speed (batch-size: {args.batch_size}):", timings=time_buffer)
        del session

    if args.tensorrt_image_model and args.device == "cuda":
        from cn_clip.deploy.tensorrt_utils import TensorRTModel
        print(f'Prepare the TensorRT image model from {args.tensorrt_image_model}')
        trt_model = TensorRTModel(args.tensorrt_image_model)
        for i in range(args.warmup):
            trt_output = trt_model(inputs={"image": image})
        time_buffer = list()
        print("Forward the TensorRT image model...")
        for i in range(args.n):
            with track_infer_time(time_buffer):
                trt_output = trt_model(inputs={"image": image})
        print_timings(name=f"TensorRT image inference speed (batch-size: {args.batch_size}):", timings=time_buffer)
        del trt_model


    # test the image feature extraction
    print("Begin the text feature extraction speed test...")
    if args.pytorch_ckpt:
        print(f"Prepare the Pytorch model from {args.pytorch_ckpt}")
        pt_model = prepare_pytorch_model(args)

        for i in range(args.warmup):
            pytorch_output = pt_model(image=None, text=text)
        print("Forward the Pytorch text model...")
        time_buffer = list()
        for i in range(args.n):
            with track_infer_time(time_buffer):
                pytorch_output = pt_model(image=None, text=text)
        print_timings(name=f"Pytorch text inference speed (batch-size: {args.batch_size}):", timings=time_buffer)
        del pt_model

    if args.onnx_text_model:
        import onnxruntime
        print(f'Prepare the ONNX text model from {args.onnx_text_model}')
        sess_options = onnxruntime.SessionOptions()
        run_options = onnxruntime.RunOptions()
        run_options.log_severity_level = 2
        session = onnxruntime.InferenceSession(args.onnx_text_model,
                                                sess_options=sess_options,
                                                providers=[onnx_execution_provider_map[args.device]]
                                                )
        for i in range(args.warmup):
            onnx_output = session.run(["unnorm_text_features"], {"text": text.cpu().numpy()})
        print("Forward the ONNX text model...")
        time_buffer = list()
        for i in range(args.n):
            with track_infer_time(time_buffer):
                onnx_output = session.run(["unnorm_text_features"], {"text": text.cpu().numpy()})
        print_timings(name=f"ONNX text inference speed (batch-size: {args.batch_size}):", timings=time_buffer)
        del session

    if args.tensorrt_text_model and args.device == "cuda":
        from cn_clip.deploy.tensorrt_utils import TensorRTModel
        print(f'Prepare the TensorRT text model from {args.tensorrt_text_model}')        
        trt_model = TensorRTModel(args.tensorrt_text_model)
        for i in range(args.warmup):
            trt_output = trt_model(inputs={'text': text})
        print("Forward the TensorRT text model...")
        time_buffer = list()
        for i in range(args.n):
            with track_infer_time(time_buffer):
                trt_output = trt_model(inputs={'text': text})
        print_timings(name=f"TensorRT text inference speed (batch-size: {args.batch_size}):", timings=time_buffer)
        del trt_model
    
    print("Done!")