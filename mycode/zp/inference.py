import json
import joblib
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import tensorflow as tf

from config import parser
from data_helper import FeatureParser
from model import MultiModal

valid = True
def main():
    args = parser.parse_args()
    feature_parser = FeatureParser(args)

    if valid:
        files = args.validfile
        args.output_json = 'valid.json'
        args.output_zip = 'valid.zip'
        two_stage_file = 'pkl/valid_two_stage_input.pkl'
    else:
        files = args.test_a_file
        args.output_json = 'result.json'
        args.output_zip = 'result.zip'
        two_stage_file = 'pkl/test_two_stage_input.pkl'
    
    dataset = feature_parser.create_dataset(files, training=False, batch_size=args.test_batch_size)
    # print(len(dataset))
    model = MultiModal(args)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(args.ckpt_file)
    print(f"Restored from {args.ckpt_file}")

    vid_embedding = {}
    two_stage_input = {}
    batch_count = 0
    for batch in dataset:
        _, embeddings, bert_embedding, vision_embedding = model(batch, training=False)
        vids = batch['vid'].numpy().astype(str)
        batch_count += 1
        embeddings = embeddings.numpy()
        bert_embeddings = bert_embedding.numpy()
        vision_embeddings = vision_embedding.numpy()
        
        for vid, embedding in zip(vids, embeddings):
            vid_embedding[vid] = embedding.tolist()
        
        for vid, embedding in zip(vids, embeddings):
            two_stage_input[vid] = []

        for vid, embedding in zip(vids, embeddings):
            two_stage_input[vid].append(embedding)
            
        for vid, embedding in zip(vids, bert_embeddings):
            two_stage_input[vid].append(embedding)
        for vid, embedding in zip(vids, vision_embeddings):
            two_stage_input[vid].append(embedding)
              
    print("batch_count:", batch_count)
    with open(args.output_json, 'w') as f:
        json.dump(vid_embedding, f)
    with ZipFile(args.output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(args.output_json)

    joblib.dump(two_stage_input, two_stage_file)


if __name__ == '__main__':
    main()
