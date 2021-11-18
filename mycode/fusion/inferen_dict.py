import json
import joblib
from zipfile import ZIP_DEFLATED, ZipFile
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from config import parser
from data_helper import FeatureParser
from model import MultiModal

valid = False
def main():
    args = parser.parse_args()
    feature_parser = FeatureParser(args)

    if valid:
        files = args.validfile
        args.output_json = 'valid.json'
        args.output_zip = 'valid.zip'
        two_stage_file = 'pkl/valid_two_stagedict_input.pkl'
    else:
        files = args.test_a_file
        args.output_json = 'result.json'
        args.output_zip = 'result.zip'
        two_stage_file = 'pkl/test_two_stagedict_input.pkl'
    
    dataset = feature_parser.create_dataset(files, training=False, batch_size=args.test_batch_size)
    # print(len(dataset))
    model = MultiModal(args)
    checkpoint = tf.train.Checkpoint(model=model)
    args.ckpt_file = ''
    checkpoint.restore(args.ckpt_file)
    print(f"Restored from {args.ckpt_file}")

    vid_embedding = {}
    # two_stage_input = {}
    bert_dict = {}
    frame_dict = {}
    fusion_dict = {}

    tag_input = {}
    batch_count = 0
    for batch in tqdm(dataset):
        _, embeddings, bert_embedding, vision_embedding = model(batch, training=False)
        vids = batch['vid'].numpy().astype(str)
        labels = batch['labels'].numpy().astype(str)
        
        batch_count += 1
        embeddings = embeddings.numpy()
        bert_embeddings = bert_embedding.numpy()
        vision_embeddings = vision_embedding.numpy()

        for vid, embedding in zip(vids, embeddings):
            fusion_dict[vid] = embedding
        for vid, embedding in zip(vids, bert_embeddings):
            bert_dict[vid] = embedding
        for vid, embedding in zip(vids, vision_embeddings):
            frame_dict[vid] = embedding
              
    print("batch_count:", batch_count)
    with open(args.output_json, 'w') as f:
        json.dump(vid_embedding, f)
    with ZipFile(args.output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(args.output_json)
    two_stage_input = [fusion_dict, bert_dict, frame_dict]
    joblib.dump(two_stage_input, two_stage_file)


if __name__ == '__main__':
    main()
