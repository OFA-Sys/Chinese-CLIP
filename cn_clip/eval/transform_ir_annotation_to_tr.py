# -*- coding: utf-8 -*-
from tqdm import tqdm
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="Input path of text-to-image Jsonl annotation file."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    t2i_record = dict()

    with open(args.input, "r", encoding="utf-8") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            text_id = obj['text_id']
            image_ids = obj['image_ids']
            for image_id in image_ids:
                if image_id not in t2i_record:
                    t2i_record[image_id] = []
                t2i_record[image_id].append(text_id)
    
    with open(args.input.replace(".jsonl", "") + ".tr.jsonl", "w", encoding="utf-8") as fout:
        for image_id, text_ids in t2i_record.items():
            out_obj = {"image_id": image_id, "text_ids": text_ids}
            fout.write("{}\n".format(json.dumps(out_obj)))
    
    print("Done!")