import logging
import os
from pprint import pprint

import tensorflow as tf

from config import parser
from data_helper import create_datasets
from metrics import Recorder
from model import MultiModal
from util import test_spearmanr

def simcse_loss(h1, h2, y, sup=False, temp=0.1):
    """用于SimCSE训练的loss
    """
    # 构造标签
    if not sup:
        idxs = tf.range(0, tf.shape(y)[0])

        idxs_1 = idxs[None, :]
        idxs_2 = idxs[:, None]
        y_true = tf.equal(idxs_1, idxs_2)
        y_true = tf.cast(y_true, tf.keras.backend.floatx())
    else:
        y =  tf.cast(y, tf.keras.backend.floatx())
        y_true = tf.dot(y, tf.transpose(y))
        y_true = y_true > 1
        y_true = tf.cast(y_true, tf.keras.backend.floatx())
    # 计算相似度
    h1 = tf.math.l2_normalize(h1, axis=1)
    h2 = tf.math.l2_normalize(h2, axis=1)
    
    similarities = tf.keras.backend.dot(h1, tf.transpose(h2)) / temp
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, similarities)
    return tf.reduce_mean(loss)

def train(args):
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_datasets(args)
    # 2. build model
    model = MultiModal(args)
    # 3. save checkpoints
    checkpoint = tf.train.Checkpoint(model=model, step=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.savedmodel_path, args.max_to_keep)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        logging.info("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")
    # 4. create loss_object and recorders
    loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_recorder, val_recorder = Recorder(), Recorder()

    # 5. define train and valid step function
    @tf.function
    def train_step(inputs):
        labels = inputs['labels']
        with tf.GradientTape() as tape:
            predictions, final_embedding, bert_embedding, vision_embedding_nextvlad, vision_embedding_netrvlad, vision_embedding_softdbow, vision_embedding_netfv = model(inputs, training=True)
            loss_tag = loss_object(labels, predictions) * labels.shape[-1] * 0.6
            loss_bert_nextvlad = simcse_loss(bert_embedding, vision_embedding_nextvlad, labels)
            loss_bert_netrvlad = simcse_loss(bert_embedding, vision_embedding_netrvlad, labels)
            loss_bert_softdbow = simcse_loss(bert_embedding, vision_embedding_softdbow, labels)
            loss_bert_netfv = simcse_loss(bert_embedding, vision_embedding_netfv, labels)
            loss = loss_tag+loss_bert_nextvlad+loss_bert_netrvlad+loss_bert_softdbow+loss_bert_netfv
        gradients = tape.gradient(loss, model.get_variables())
        model.optimize(gradients)
        train_recorder.record(loss, labels, predictions)

    @tf.function
    def val_step(inputs):
        vids = inputs['vid']
        labels = inputs['labels']
        predictions, final_embedding, bert_embedding, vision_embedding_nextvlad, vision_embedding_netrvlad, vision_embedding_softdbow, vision_embedding_netfv = model(inputs, training=False)
        loss_tag = loss_object(labels, predictions) * labels.shape[-1]
        loss_bert_nextvlad = simcse_loss(bert_embedding, vision_embedding_nextvlad, labels)
        loss_bert_netrvlad = simcse_loss(bert_embedding, vision_embedding_netrvlad, labels)
        loss_bert_softdbow = simcse_loss(bert_embedding, vision_embedding_softdbow, labels)
        loss_bert_netfv = simcse_loss(bert_embedding, vision_embedding_netfv, labels)
        loss = loss_tag+loss_bert_nextvlad+loss_bert_netrvlad+loss_bert_softdbow+loss_bert_netfv
        val_recorder.record(loss, labels, predictions)
        return vids, predictions, final_embedding, bert_embedding, vision_embedding_nextvlad, vision_embedding_netrvlad, vision_embedding_softdbow, vision_embedding_netfv

    # 6. training
    for epoch in range(args.start_epoch, args.epochs):
        for train_batch in train_dataset:
            checkpoint.step.assign_add(1)
            step = checkpoint.step.numpy()
            if step > args.total_steps:
                break
            train_step(train_batch)
            if step % args.print_freq == 0:
                train_recorder.log(epoch, step)
                train_recorder.reset()

            # 7. validation
            if step % args.eval_freq == 0:
                if args.VAL==False:
                    vid_embedding = {}
                    for val_batch in val_dataset:
                        vids, predictions, final_embedding, bert_embedding, vision_embedding_nextvlad, vision_embedding_netrvlad, vision_embedding_softdbow, vision_embedding_netfv = val_step(val_batch)
                        for vid, embedding in zip(vids.numpy(), final_embedding.numpy()):
                            vid = vid.decode('utf-8')
                            vid_embedding[vid] = embedding
                    # 8. test spearman correlation
                    spearmanr = test_spearmanr(vid_embedding, args.annotation_file)
                    val_recorder.log(epoch, step, prefix='Validation result is: ', suffix=f', spearmanr {spearmanr:.4f}')
                    val_recorder.reset()

                    # 9. save checkpoints
                    if spearmanr > 0.45:
                        checkpoint_manager.save(checkpoint_number=step)
                else:
                    predictions_dict = {}
                    final_embedding_dict = {}
                    bert_embedding_dict = {}
                    vision_embedding_nextvlad_dict = {}
                    vision_embedding_netrvlad_dict = {}
                    vision_embedding_softdbow_dict = {}
                    vision_embedding_netf_dict = {}
                    for val_batch in val_dataset:
                        vids, predictions, final_embedding, bert_embedding, vision_embedding_nextvlad, vision_embedding_netrvlad, vision_embedding_softdbow, vision_embedding_netfv = val_step(val_batch)
                        for vid, embedding in zip(vids.numpy(), final_embedding.numpy()):
                            vid = vid.decode('utf-8')
                            predictions_dict[vid] = predictions
                            final_embedding_dict[vid] = final_embedding
                            bert_embedding_dict[vid] = bert_embedding
                            vision_embedding_nextvlad_dict[vid] = vision_embedding_nextvlad
                            vision_embedding_netrvlad_dict[vid] = vision_embedding_netrvlad
                            vision_embedding_softdbow_dict[vid] = vision_embedding_softdbow
                            vision_embedding_netf_dict[vid] = vision_embedding_netf
                    with open('predictions_val', 'w') as f:
                        json.dump(predictions_dict, f)
                    with open('final_embedding_val', 'w') as f:
                        json.dump(final_embedding_dict, f)
                    with open('bert_embedding_val', 'w') as f:
                        json.dump(bert_embedding_dict, f)
                    with open('vision_embedding_nextvlad_val', 'w') as f:
                        json.dump(vision_embedding_nextvlad_dict, f)
                    with open('vision_embedding_netrvlad_val', 'w') as f:
                        json.dump(vision_embedding_netrvlad_dict, f)
                    with open('vision_embedding_softdbow_val', 'w') as f:
                        json.dump(vision_embedding_softdbow_dict, f)
                    with open('vision_embedding_netf_val', 'w') as f:
                        json.dump(vision_embedding_netf_dict, f)

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    args = parser.parse_args()

    if not os.path.exists(args.savedmodel_path):
        os.makedirs(args.savedmodel_path)

    pprint(vars(args))
    train(args)


if __name__ == '__main__':
    main()
