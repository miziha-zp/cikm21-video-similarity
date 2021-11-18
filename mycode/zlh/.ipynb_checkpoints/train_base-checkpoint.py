import logging
import os
from pprint import pprint

import tensorflow as tf

from config_base import parser
from data_helper_base import create_datasets
from metrics_base import Recorder
from model_transFrame_base import MultiModal
from util import test_spearmanr

weight_1 = 0.20
weight_2 = 0.15
weight_3 = 0.65
alpha = 100.0

def cosine_distance(x1, x2):
    y_pred_1 = tf.nn.l2_normalize(x1, axis = 1)
    y_pred_2 = tf.nn.l2_normalize(x2, axis = 1)
    similarities = tf.matmul(y_pred_1, tf.transpose(y_pred_2))
    similarities = tf.cast(tf.linalg.tensor_diag_part(similarities), tf.float32)
#     cosin = 1 - tf.acos(cosin)/3.1415927
    return similarities

def simcse_loss(y_pred,y_ture):
    idxs = tf.range(0, tf.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = tf.equal(idxs_1, idxs_2)
    y_true = tf.cast(y_true, tf.float32)
    # # 计算相似度
    y_pred = tf.nn.l2_normalize(y_pred, axis = 1)
    similarities = tf.matmul(y_pred, tf.transpose(y_pred))
    similarities = tf.cast(similarities, tf.float32)
    similarities = similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = tf.keras.backend.categorical_crossentropy(y_true, similarities, from_logits=True)
    return tf.reduce_mean(loss)

def maskmeanpooling(x, mask):
    # 有数据的位置为1 无数据的位置为0
    mask = tf.cast(mask, tf.float32)  # mask (batch, time)
    mask = tf.expand_dims(mask, -1)# mask (batch, time, 1)
    a = x * mask
    b = tf.reduce_sum(a, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-10)
    return b

def train(args):
    # 1. create dataset and set num_labels to args
    train_dataset_1, val_dataset_1, train_dataset_2, val_dataset_2 = create_datasets(args)
    # 2. build model
    model = MultiModal(args)
    # 3. save checkpoints
    checkpoint = tf.train.Checkpoint(model=model, step=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.savedmodel_path, args.max_to_keep)
    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    if checkpoint_manager.latest_checkpoint:
        logging.info("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")
    # 4. create loss_object and recorders
    loss_object_tag = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_recorder, val_recorder = Recorder(), Recorder()
    # 5. define train and valid step function
    @tf.function
    def train_step(train_dataset_1,train_dataset_2):
        labels_1 = train_dataset_1['labels']
        labels_2 = train_dataset_2['labels']
        with tf.GradientTape() as tape:
            predictions_1, emb_1, _, _, _, _ = model(train_dataset_1, training=True)
            predictions_2, emb_2, _, _, _, _ = model(train_dataset_1, training=True)
            loss_1 = loss_object_tag(labels_1, predictions_1) * labels_1.shape[-1]  # convert mean back to sum
            loss_2 = loss_object_tag(labels_2, predictions_2) * labels_2.shape[-1]  # convert mean back to sum
            emb_all = tf.reshape(tf.concat([emb_1, emb_2],axis=1),(-1,emb_1.shape[-1]))
            loss_3 = simcse_loss(emb_all, emb_all)# 
            loss = (loss_1 * weight_1 + loss_2 * weight_2) + (loss_3 * weight_3)*alpha
        gradients = tape.gradient(loss, model.get_variables())
        model.optimize(gradients)
        train_recorder.record(loss, labels_1, predictions_1, loss_1, loss_2, loss_3)

    @tf.function
    def val_step(train_dataset_1,train_dataset_2):
        vids = train_dataset_1['vid']
        labels_1 = train_dataset_1['labels']
        labels_2 = train_dataset_2['labels']
        predictions_1, embeddings_1, _, _, _, _ = model(train_dataset_1, training=False)
        predictions_2, embeddings_2, _, _, _, _ = model(train_dataset_2, training=False)
        loss_1 = loss_object_tag(labels_1, predictions_1) * labels_1.shape[-1]  # convert mean back to sum
        loss_2 = loss_object_tag(labels_2, predictions_2) * labels_2.shape[-1]  # convert mean back to sum
        emb_all = tf.reshape(tf.concat([embeddings_1, embeddings_2],axis=1),(-1,embeddings_1.shape[-1]))
        loss_3 = simcse_loss(emb_all, emb_all)# 
        loss = (loss_1 * weight_1 + loss_2 * weight_2)*alpha + loss_3 * weight_3
        val_recorder.record(loss, labels_1, predictions_1, loss_1, loss_2, loss_3)
        return vids, embeddings_1

    # 6. training
    total_index = 0
    best_spearmanr = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        for train_batch_1,train_batch_2 in zip(train_dataset_1,train_dataset_2):
            checkpoint.step.assign_add(1)
            step = checkpoint.step.numpy()
            if step > args.total_steps:
                break
            train_step(train_batch_1,train_batch_2)
            if total_index==0:
                print(model.summary())
                total_index+=1
            if step % args.print_freq == 0:
                train_recorder.log(epoch, step)
                train_recorder.reset()
            # 7. validation
            if step % args.eval_freq == 0:
                vid_embedding = {}
                for val_batch_1,val_batch_2 in zip(val_dataset_1,val_dataset_2):
                    vids, embeddings = val_step(val_batch_1,val_batch_2)
                    for vid, embedding in zip(vids.numpy(), embeddings.numpy()):
                        vid = vid.decode('utf-8')
                        vid_embedding[vid] = embedding
                # 8. test spearman correlation
                spearmanr = test_spearmanr(vid_embedding, args.annotation_file)
                val_recorder.log(epoch, step, prefix='Validation result is: ', suffix=f', spearmanr {spearmanr:.4f}')
                val_recorder.reset()
                if spearmanr>best_spearmanr:
                    best_spearmanr = spearmanr
                    print("find new best valid spearmanr:",best_spearmanr)
                    # 9. save checkpoints
                    checkpoint_manager.save(checkpoint_number=step)
                    


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
