# -*- coding: utf-8 -*-
'''
This scripts performs kNN search on inferenced image and text features (on single-GPU) and outputs text-to-image prediction file for evaluation.
'''

import argparse
import numpy
from tqdm import tqdm
import json

import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-feats', 
        type=str, 
        required=True,
        help="Specify the path of image features."
    )  
    parser.add_argument(
        '--text-feats', 
        type=str, 
        required=True,
        help="Specify the path of text features."
    )      
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=10,
        help="Specify the k value of top-k predictions."
    )   
    parser.add_argument(
        '--eval-batch-size', 
        type=int, 
        default=32768,
        help="Specify the image-side batch size when computing the inner products, default to 8192"
    )    
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help="Specify the output jsonl prediction filepath."
    )         
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    print("Begin to load image features...")
    image_ids = []
    image_feats = []
    with open(args.image_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            image_ids.append(obj['image_id'])
            image_feats.append(obj['feature'])
    image_feats_array = np.array(image_feats, dtype=np.float32)
    print("Finished loading image features.")

    print("Begin to compute top-{} predictions for texts...".format(args.top_k))
    with open(args.output, "w") as fout:
        with open(args.text_feats, "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())
                text_id = obj['text_id']
                text_feat = obj['feature']
                score_tuples = []
                text_feat_tensor = torch.tensor([text_feat], dtype=torch.float).cuda() # [1, feature_dim]
                idx = 0
                while idx < len(image_ids):
                    img_feats_tensor = torch.from_numpy(image_feats_array[idx : min(idx + args.eval_batch_size, len(image_ids))]).cuda() # [batch_size, feature_dim]
                    batch_scores = text_feat_tensor @ img_feats_tensor.t() # [1, batch_size]
                    for image_id, score in zip(image_ids[idx : min(idx + args.eval_batch_size, len(image_ids))], batch_scores.squeeze(0).tolist()):
                        score_tuples.append((image_id, score))
                    idx += args.eval_batch_size
                top_k_predictions = sorted(score_tuples, key=lambda x:x[1], reverse=True)[:args.top_k]
                fout.write("{}\n".format(json.dumps({"text_id": text_id, "image_ids": [entry[0] for entry in top_k_predictions]})))
    
    print("Top-{} predictions are saved in {}".format(args.top_k, args.output))
    print("Done!")
