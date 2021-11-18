import logging
import os
from pprint import pprint

import tensorflow as tf

from config import parser
from data_helper import create_datasets
from metrics import Recorder
from model import MultiModal
from util import test_spearmanr


def train(args):
    # 0. 多卡并行
    strategy = tf.distribute.MirroredStrategy()
    GLOBAL_BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync # Global batch size
    BUFFER_SIZE = args.batch_size * strategy.num_replicas_in_sync * 16 # Buffer size for data loader
    
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_datasets(args)
    dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    dist_val_dataset = strategy.experimental_distribute_dataset(val_dataset)

    with strategy.scope():
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
        def compute_loss(logits, labels):
            per_example_loss = loss_object(y_true=labels, y_pred=logits) * labels.shape[-1] # convert mean back to sum
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        train_recorder, val_recorder = Recorder(), Recorder()
        
    # 5. define train and valid step function
    with strategy.scope():
        @tf.function
        def train_step(inputs):
            labels = inputs['labels']
            with tf.GradientTape() as tape:
                predictions, _ = model(inputs, training=True)
                loss = compute_loss(logits=predictions, labels=labels)  
            gradients = tape.gradient(loss, model.get_variables())
            model.optimize(gradients)
            train_recorder.record(loss, labels, predictions)
            
    with strategy.scope():   
        @tf.function
        def val_step(inputs):
            vids = inputs['vid']
            labels = inputs['labels']
            predictions, embeddings = model(inputs, training=False)
            loss = compute_loss(logits=predictions, labels=labels)  
            val_recorder.record(loss, labels, predictions)
            return vids, embeddings

    # 6. training
    with strategy.scope():
        for epoch in range(args.start_epoch, args.epochs):
            for train_batch in dist_train_dataset:
                checkpoint.step.assign_add(1)
                step = checkpoint.step.numpy()
                if step > args.total_steps:
                    break
                strategy.run(train_step, args=(train_batch,))
                if step % args.print_freq == 0:
                    train_recorder.log(epoch, step)
                    train_recorder.reset()

                # 7. validation
                if step % args.eval_freq == 0:
                    vid_embedding = {}
                    for val_batch in dist_val_dataset:
                        vid_dist, embedding_dist = strategy.run(val_step, args=(val_batch,))
                        for vids, embeddings in zip(vid_dist.values, embedding_dist.values):
                            for vid, embedding in zip(vids.numpy(), embeddings.numpy()):
                                vid = vid.decode('utf-8')
                                vid_embedding[vid] = embedding
                    
                    # 8. test spearman correlation
                    spearmanr = test_spearmanr(vid_embedding, args.annotation_file)
                    val_recorder.log(epoch, step, prefix='Validation result is: ', suffix=f', spearmanr {spearmanr:.4f}')
                    val_recorder.reset()  
                    
                    # 9. save checkpoints
                    if spearmanr > 0.45:
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
