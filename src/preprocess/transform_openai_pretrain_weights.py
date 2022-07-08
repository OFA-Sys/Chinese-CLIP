# -*- coding: utf-8 -*-
'''
This script transforms the format of openai pretrained CLIP weights from a JIT-loaded model into a state-dict.
'''

import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-ckpt-path", type=str, required=True, help="specify the path of the original openai checkpoint")
    parser.add_argument("--new-ckpt-path", type=str, default=None, help="specify the path of the transformed checkpoint")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")    

    assert os.path.exists(args.raw_ckpt_path), "The raw ckpt path does not exist!"
    if args.new_ckpt_path is None:
        ext_len = len(os.path.splitext(args.raw_ckpt_path)[-1])
        args.new_ckpt_path = "{}.state_dict{}".format(args.raw_ckpt_path[:-ext_len], args.raw_ckpt_path[-ext_len:])
    
    model = torch.jit.load(args.raw_ckpt_path, map_location='cpu')

    sd = model.state_dict()

    torch.save(sd, args.new_ckpt_path)
    print("Transformed openai ckpt {} to {}!".format(args.raw_ckpt_path, args.new_ckpt_path))
    print("Done!")