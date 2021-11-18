import json
import joblib
from zipfile import ZIP_DEFLATED, ZipFile
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from config_base import parser
from data_helper import FeatureParser
from model_transFrame_base import MultiModal

def main():
    args = parser.parse_args([])
    feature_parser = FeatureParser(args)

    if valid:
        files = args.validfile
        args.output_json = 'valid.json'
        args.output_zip = 'valid.zip'
        two_stage_file = './embedding/valid_two_stagedict_input_zlh1002.pkl'
    else:
        files = args.test_a_file
        args.output_json = 'result.json'
        args.output_zip = 'result.zip'
        two_stage_file = './embedding/test_two_stagedict_input_zlh1002.pkl'
    
    dataset = feature_parser.create_dataset(files, training=False, batch_size=args.test_batch_size)
    # print(len(dataset))
    model = MultiModal(args)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.savedmodel_path, args.max_to_keep)
    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial() # 最后一次checkpoint
    print("Restored from {}".format(checkpoint_manager.latest_checkpoint))

    vid_embedding = {}
    # two_stage_input = {}
    bert_dict_title = {}
    bert_dict_asr = {}
    frame_dict = {}
    fusion_dict = {}
    prediction_dict = {}
    tag_input = {}
    batch_count = 0
    for batch in tqdm(dataset):
        prediction, embeddings,emb_1,emb_2,emb_3,_ = model(batch, training=False)
        vids = batch['vid'].numpy().astype(str)
        labels = batch['labels'].numpy().astype(str)
        
        batch_count += 1
        embeddings = embeddings.numpy().astype(np.float16)
        for vid, embedding in zip(vids, embeddings):
            fusion_dict[vid] = embedding
        embeddings = emb_1.numpy().astype(np.float16)
        for vid, embedding in zip(vids, embeddings):
            bert_dict_title[vid] = embedding
        embeddings = emb_2.numpy().astype(np.float16)
        for vid, embedding in zip(vids, embeddings):
            bert_dict_asr[vid] = embedding
        embeddings = emb_3.numpy().astype(np.float16)
        for vid, embedding in zip(vids, embeddings):
            frame_dict[vid] = embedding
        embeddings = prediction.numpy().astype(np.float16)
        for vid, embedding in zip(vids, embeddings):
            prediction_dict[vid] = embedding
    print("batch_count:", batch_count)
    with open(args.output_json, 'w') as f:
        json.dump(vid_embedding, f)
    with ZipFile(args.output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(args.output_json)
    two_stage_input = [prediction_dict,fusion_dict,bert_dict_title,bert_dict_asr,frame_dict]
    joblib.dump(two_stage_input, two_stage_file)


if __name__ == '__main__':
    valid = True
    main()
    valid = False
    main()