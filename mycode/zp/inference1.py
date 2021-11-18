import json
import joblib
from zipfile import ZIP_DEFLATED, ZipFile
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from config_simclr import parser
from data_helper import FeatureParser
from model2 import MultiModal


def main():
    args = parser.parse_args()
    feature_parser = FeatureParser(args)
    print(args.inference_name)
    valid = False if args.inference_name == 'test' else True
    if valid:
        files = args.validfile
        args.output_json = 'valid.json'
        args.output_zip = 'valid.zip'
        two_stage_file = '../embedding/roberta_valid_seq.pkl'
    else:
        files = args.test_b_file
        args.output_json = 'result.json'
        args.output_zip = 'result.zip'
        two_stage_file = '../embedding/roberta_test_seq.pkl'
    
    dataset = feature_parser.create_dataset(files, training=False, batch_size=args.test_batch_size)
    # print(len(dataset))

    model = MultiModal(args)
    checkpoint = tf.train.Checkpoint(model=model)
    args.ckpt_file = 'save/model1/ckpt-28000'
    checkpoint.restore(args.ckpt_file)
    print(f"Restored from {args.ckpt_file}")

    vid_embedding = {}
    # two_stage_input = {}
    bert_dict = {}
    frame_dict = {}
    fusion_dict = {}
    bert_last_dict = {}
    predictions_dict = {}
    tag_input = {}
    batch_count = 0
    for batch in tqdm(dataset):
        predictions, embeddings, bert_embedding, allbert_out, vision_embedding = model(batch, training=False)
        vids = batch['vid'].numpy().astype(str)
        labels = batch['labels'].numpy().astype(str)
#         print('-'*100)
#         print(bert_embedding2[0].shape) #(576, 32, 768)
#         print(bert_embedding2[1].shape) #(576, 768)
#         for i in range(13):
#             print(i, bert_embedding2[2][i].shape) #(576, 32, 768)
#         assert bert_embedding2[2][-1] == bert_embedding2[1]

#         last = bert_embedding2[0].numpy().astype(np.float16)
#         print('-'*100)
        batch_count += 1
        embeddings = embeddings.numpy().astype(np.float16)
        bert_embeddings = bert_embedding.numpy().astype(np.float16)
        bert_last_embeddings = allbert_out[0].numpy().astype(np.float16)
        vision_embeddings = vision_embedding.numpy().astype(np.float16)
        predictions = predictions.numpy().astype(np.float16)
        
        for vid, embedding in zip(vids, embeddings):
            fusion_dict[vid] = embedding
        for vid, embedding in zip(vids, bert_last_embeddings):
            bert_last_dict[vid] = embedding
        for vid, embedding in zip(vids, bert_embeddings):
            bert_dict[vid] = embedding
        for vid, embedding in zip(vids, predictions):
            predictions_dict[vid] = embedding
        for vid, embedding in zip(vids, vision_embeddings):
            frame_dict[vid] = embedding

            
    print("batch_count:", batch_count)
    with open(args.output_json, 'w') as f:
        json.dump(vid_embedding, f)
    with ZipFile(args.output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(args.output_json)
    two_stage_input = [fusion_dict, bert_dict, frame_dict, predictions_dict, bert_last_dict]
    joblib.dump(two_stage_input, two_stage_file)


if __name__ == '__main__':
    main()
