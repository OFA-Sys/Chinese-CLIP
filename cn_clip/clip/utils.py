# Code modified from https://github.com/openai/CLIP

import json
import os
from pathlib import Path
from typing import Union, List

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode

from cn_clip.clip import _tokenizer
from cn_clip.clip.model import convert_weights, CLIP, restore_model

__all__ = ["load", "tokenize", "available_models", "image_transform", "load_from_name"]

_MODELSCOPE_ORG = "AI-ModelScope"
_HUGGINGFACE_ORG = "OFA-Sys"

_MODELS = {
    "ViT-B-16": ("chinese-clip-vit-base-patch16", "clip_cn_vit-b-16.pt"),
    "ViT-L-14": ("chinese-clip-vit-large-patch14", "clip_cn_vit-l-14.pt"),
    "ViT-L-14-336": ("chinese-clip-vit-large-patch14-336px", "clip_cn_vit-l-14-336.pt"),
    "ViT-H-14": ("chinese-clip-vit-huge-patch14", "clip_cn_vit-h-14.pt"),
    "RN50": ("chinese-clip-rn50", "clip_cn_rn50.pt"),
}

_MODEL_INFO = {
    "ViT-B-16": {
        "struct": "ViT-B-16@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "ViT-L-14": {
        "struct": "ViT-L-14@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "ViT-L-14-336": {
        "struct": "ViT-L-14-336@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 336
    },
    "ViT-H-14": {
        "struct": "ViT-H-14@RoBERTa-wwm-ext-large-chinese",
        "input_resolution": 224
    },
    "RN50": {
        "struct": "RN50@RBT3-chinese",
        "input_resolution": 224
    },
}


def _download(modelname: str, root: str, use_modelscope: bool = False):
    os.makedirs(root, exist_ok=True)

    # this a private function and the only caller has checked the model exsits
    reponame, filename = _MODELS[modelname]

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    if use_modelscope:
        try:
            from modelscope.hub.file_download import model_file_download
        except ImportError as _:
            raise RuntimeError(
                "Ckpt download requires `modelscope`. "
                "Please install `modelscope` or download the ckpt manually and "
                "provide the local path so that we can continue."
            )

        local_path = model_file_download(
            model_id=_MODELSCOPE_ORG + "/" + reponame,
            file_path=filename,
            local_dir=root,
        )
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as _:
            raise RuntimeError(
                "Ckpt download requires `huggingface_hub`. "
                "Please install `huggingface_hub` or download the ckpt manually and "
                "provide the local path so that we can continue."
            )

        local_path = hf_hub_download(
            _HUGGINGFACE_ORG + "/" + reponame, filename=filename, local_dir=root
        )

    return local_path


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load_from_name(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                   download_root: str = None, vision_model_name: str = None, text_model_name: str = None, input_resolution: int = None, use_modelscope: bool = False):
    if name in _MODELS:
        model_path = _download(name, download_root or os.path.expanduser("~/.cache/clip"), use_modelscope)
        model_name, model_input_resolution = _MODEL_INFO[name]['struct'], _MODEL_INFO[name]['input_resolution']
    elif os.path.isfile(name):
        assert vision_model_name and text_model_name and input_resolution, "Please specify specific 'vision_model_name', 'text_model_name', and 'input_resolution'"
        model_path = name
        model_name, model_input_resolution = f'{vision_model_name}@{text_model_name}', input_resolution
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        # loading saved checkpoint
        checkpoint = torch.load(opened_file, map_location="cpu")

    model = create_model(model_name, checkpoint)
    if str(device) == "cpu":
        model.float()
    else:
        model.to(device)
    return model, image_transform(model_input_resolution)


def load(model, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", clip_path=None,
         bert_path=None, use_flash_attention=False):
    """Load CLIP and BERT model weights
    """

    bert_state_dict = torch.load(bert_path, map_location="cpu") if bert_path else None
    clip_state_dict = torch.load(clip_path, map_location="cpu") if clip_path else None

    restore_model(model, clip_state_dict, bert_state_dict, use_flash_attention).to(device)

    if str(device) == "cpu":
        model.float()
    return model


def tokenize(texts: Union[str, List[str]], context_length: int = 52) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 52 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[
                                                        :context_length - 2] + [_tokenizer.vocab['[SEP]']])

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(image_size=224):
    transform = Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform


def create_model(model_name, checkpoint=None):
    vision_model, text_model = model_name.split('@')
    # Initialize the model.
    vision_model_config_file = Path(
        __file__).parent / f"model_configs/{vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)

    text_model_config_file = Path(
        __file__).parent / f"model_configs/{text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        for k, v in json.load(ft).items():
            model_info[k] = v
    if isinstance(model_info['vision_layers'], str):
        model_info['vision_layers'] = eval(model_info['vision_layers'])
    print('Model info', model_info)
    model = CLIP(**model_info)
    convert_weights(model)
    if checkpoint:
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
        model.load_state_dict(sd)
    return model
