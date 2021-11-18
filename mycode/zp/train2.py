import logging
import os
from pprint import pprint
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from config_simclr import parser
from data_helper import create_datasets
from metrics import Recorder
from model import MultiModal
from util import test_spearmanr

def tf_contrastive_loss(out,out_aug,batch_size=192,hidden_norm=False,temperature=1.0):
    if hidden_norm:
        out=tf.nn.l2_normalize(out,-1)
        out_aug=tf.nn.l2_normalize(out_aug,-1)
    INF = np.inf
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2) #[batch_size,2*batch_size]
    masks = tf.one_hot(tf.range(batch_size), batch_size) #[batch_size,batch_size]
    logits_aa = tf.matmul(out, out, transpose_b=True) / temperature #[batch_size,batch_size]
    logits_bb = tf.matmul(out_aug, out_aug, transpose_b=True) / temperature #[batch_size,batch_size]
    logits_aa = logits_aa - masks * INF # remove the same samples in out
    logits_bb = logits_bb - masks * INF # remove the same samples in out_aug
    logits_ab = tf.matmul(out, out_aug, transpose_b=True) / temperature
    logits_ba = tf.matmul(out_aug, out, transpose_b=True) / temperature
    loss_a = tf.losses.categorical_crossentropy(
        labels, tf.concat([logits_ab, logits_aa], 1), from_logits=True)
    loss_b = tf.losses.categorical_crossentropy(
        labels, tf.concat([logits_ba, logits_bb], 1), from_logits=True)
    loss=loss_a+loss_b
    return loss,logits_ab

def simcse_loss(h1, h2, y, sup=False, temp=0.1):
    """用于SimCSE训练的loss
    """
    # 构造标签
    if not sup:
        idxs = K.arange(0, K.shape(y)[0])

        idxs_1 = idxs[None, :]
        idxs_2 = idxs[:, None]
        y_true = K.equal(idxs_1, idxs_2)
        y_true = K.cast(y_true, K.floatx())
    else:
        y =  K.cast(y, K.floatx())
        y_true = K.dot(y, K.transpose(y))
        y_true = y_true > 1
        y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    h1 = K.l2_normalize(h1, axis=1)
    h2 = K.l2_normalize(h2, axis=1)
    
    similarities = K.dot(h1, K.transpose(h2)) / temp
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)


def train(args):
    print(args)
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_datasets(args)
    # 2. build model
    model = MultiModal(args)
    # 3. save checkpoints
    checkpoint = tf.train.Checkpoint(model=model, step=tf.Variable(0))
    # checkpoint = tf.train.Checkpoint(model=model)
 
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.savedmodel_path, args.max_to_keep)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
#     checkpoint.restore(args.ckpt_file)
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
            predictions1, final_embedding1,bert_embedding1, vision_embedding1= model(inputs, training=True)
            predictions2, final_embedding2, bert_embedding2, vision_embedding2 = model(inputs, training=True)
            TEMP = 0.1
            loss_label = loss_object(labels, predictions1) * labels.shape[-1]  # convert mean back to sum
            loss_simclr1 = simcse_loss(vision_embedding1, vision_embedding2, labels, temp=TEMP) #* labels.shape[-2]
            loss_simclr2 = simcse_loss(bert_embedding1, bert_embedding2, labels, temp=TEMP) #* labels.shape[-2]
            loss_simclr3 = simcse_loss(final_embedding1, final_embedding2, labels,  temp=TEMP)#* labels.shape[-2]
            
            loss_simclr41 = simcse_loss(bert_embedding1, vision_embedding1, labels,  temp=TEMP)#* labels.shape[-2]
            loss_simclr42 = simcse_loss(vision_embedding1, bert_embedding1, labels,  temp=TEMP)#* labels.shape[-2]
            
            loss_simclr5 = simcse_loss(bert_embedding1, final_embedding1, labels,  temp=TEMP)#* labels.shape[-2]
            loss_simclr6 = simcse_loss(vision_embedding1, final_embedding1, labels,  temp=TEMP)#* labels.shape[-2]
            loss = loss_label + loss_simclr41 + loss_simclr42 + loss_simclr1
#             loss = loss_simclr1 + loss_simclr2 +loss_simclr3 +loss_simclr41 + loss_simclr42 + loss_label + loss_simclr5 + loss_simclr6
        gradients = tape.gradient(loss, model.get_variables())
        model.optimize(gradients)
        train_recorder.record(loss, labels, predictions1)

    @tf.function
    def val_step(inputs):
        vids = inputs['vid']
        labels = inputs['labels']
        predictions1, embeddings1, bert_embedding, vision_embedding= model(inputs, training=False)
        predictions2, embeddings2, bert_embedding, vision_embedding= model(inputs, training=False)
#         loss_simclr= tf_contrastive_loss(final_embedding1, final_embedding2)
        loss = simcse_loss(embeddings1, embeddings1, labels) * labels.shape[-1]  # convert mean back to sum
        val_recorder.record(loss, labels, predictions1)
        return vids, embeddings1, bert_embedding, vision_embedding

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

            if step % args.eval_freq == 0:
                vid_embedding = {}
                vid_bert_embedding = {}
                vid_vision_embedding = {}
                for val_batch in val_dataset:
                    vids, embeddings, bert_embedding, vision_embedding = val_step(val_batch)
                    for vid, embedding in zip(vids.numpy(), embeddings.numpy()):
                        vid = vid.decode('utf-8')
                        vid_embedding[vid] = embedding
                        
                    for vid, embedding in zip(vids.numpy(), bert_embedding.numpy()):
                        vid = vid.decode('utf-8')
                        vid_bert_embedding[vid] = embedding
                        
                    for vid, embedding in zip(vids.numpy(), vision_embedding.numpy()):
                        vid = vid.decode('utf-8')
                        vid_vision_embedding[vid] = embedding
                        
                        
                # 8. test spearman correlation
                
                print("vid_bert_embedding")
                spearmanr = test_spearmanr(vid_bert_embedding, args.annotation_file)
                print(f"vid_bert_embedding ====> {spearmanr:.4f}")
                print("vid_vision_embedding")
                spearmanr = test_spearmanr(vid_vision_embedding, args.annotation_file)
                print(f"vid_vision_embedding ====> {spearmanr:.4f}")
                print("vid_embedding")
                spearmanr = test_spearmanr(vid_embedding, args.annotation_file)
                print(f"vid_embedding ====> {spearmanr:.4f}\n")
                val_recorder.log(epoch, step, prefix='Validation result is: ', suffix=f', spearmanr {spearmanr:.4f}')
                val_recorder.reset()

            
                # 9. save checkpoints
                if spearmanr > 0.:
                    checkpoint_manager.save(checkpoint_number=step)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    args = parser.parse_args()
    args.savedmodel_path = 'save/model2/'

    if not os.path.exists(args.savedmodel_path):
        os.makedirs(args.savedmodel_path)

    pprint(vars(args))
    train(args)


if __name__ == '__main__':
    main()
