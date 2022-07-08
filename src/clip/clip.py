# Code modified from https://github.com/openai/CLIP

from typing import Union, List

import torch

from clip import _tokenizer
from clip.model import restore_model

__all__ = ["load", "tokenize"]


def load(model, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", clip_path=None, bert_path=None):
    """Load CLIP and BERT model weights
    """
    bert_state_dict = torch.load(bert_path, map_location="cpu") if bert_path else None
    clip_state_dict = torch.load(clip_path, map_location="cpu") if clip_path else None

    restore_model(model, clip_state_dict, bert_state_dict).to(device)

    if str(device) == "cpu":
        model.float()
    return model


def tokenize(texts: Union[str, List[str]], context_length: int = 64) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 24 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[:context_length - 2] + [_tokenizer.vocab['[SEP]']])

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result