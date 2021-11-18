import glob
import logging

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from transformers import BertTokenizer


class FeatureParser:
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        self.max_bert_length = args.bert_seq_length
        self.max_frames = args.max_frames

        self.selected_tags = set()
        with open(args.multi_label_file, encoding='utf-8') as fh:
            for line in fh:
                tag_id = int(line.strip())
                self.selected_tags.add(tag_id)
        self.num_labels = len(self.selected_tags)
        args.num_labels = self.num_labels
        logging.info('Num of selected supervised qeh tags is {}'.format(self.num_labels))
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

    def change_frames(self, frames, rate=0.75):
        ### frames(32,1536)
        list_use = np.where(np.random.random((len(frames)-2,)) > rate)[0]
        for frame_num in list_use:
            frames[frame_num] = frames[frame_num + 1]
            frames[frame_num + 2] = frames[frame_num + 1]
        return frames

    def _encode(self, text, ifc=False):
        text = text.numpy().decode(encoding='utf-8')
        if ifc:
            rate = 0.75
            s_ = ""  # 变更后的语料
            list_use = np.where(np.random.random((len(text),)) > rate)[0]
            for index, w in enumerate(list_use):
                if index == 0:
                    s_ += text[0:w]
                if index < len(list_use) - 1:
                    s_ += text[w] + text[w:int(list_use[index + 1])]
                else:
                    s_ += text[w] + text[w:]
            text = s_
        encoded_inputs = self.tokenizer(text, max_length=self.max_bert_length, padding='max_length', truncation=True)
        input_ids = encoded_inputs['input_ids']
        mask = encoded_inputs['attention_mask']
        return input_ids, mask

    def _parse_title(self, title, ifc=False):
        input_ids, mask = tf.py_function(lambda x: self._encode(x, ifc=ifc), [title], [tf.int32, tf.int32])
        input_ids.set_shape([self.max_bert_length])
        mask.set_shape([self.max_bert_length])
        return input_ids, mask

    def _parse_asr(self, asr, ifc=False):
        input_ids, mask = tf.py_function(lambda x: self._encode(x, ifc=ifc), [asr], [tf.int32, tf.int32])
        input_ids.set_shape([self.max_bert_length])
        mask.set_shape([self.max_bert_length])
        return input_ids, mask

    def _sample(self, frames, ifc=False):
        frames = frames.numpy()
        if ifc:
            frames = self.change_frames(frames)
        frames_len = len(frames)
        num_frames = min(frames_len, self.max_frames)
        num_frames = np.array([num_frames], dtype=np.int32)

        average_duration = frames_len // self.max_frames
        if average_duration == 0:
            return [frames[min(i, frames_len - 1)] for i in range(self.max_frames)], num_frames
        else:
            offsets = np.multiply(list(range(self.max_frames)), average_duration) + average_duration // 2
            return [frames[i] for i in offsets], num_frames

    def _parse_frames(self, frames, ifc=False):
        frames = tf.sparse.to_dense(frames)
        frames, num_frames = tf.py_function(lambda x: self._sample(x, ifc=ifc), [frames], [tf.string, tf.int32])
        frames_embedding = tf.map_fn(lambda x: tf.io.decode_raw(x, out_type=tf.float16), frames, dtype=tf.float16)
        frames_embedding = tf.cast(frames_embedding, tf.float32)
        frames_embedding.set_shape([self.max_frames, self.args.frame_embedding_size])
        num_frames.set_shape([1])
        return frames_embedding, num_frames

    def _parse_label(self, labels):
        tags = labels.numpy()
        # tag filtering
        tags = [tag for tag in tags if tag in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0].astype(dtype=np.int8)
        return tf.convert_to_tensor(multi_hot)

    def _parse_labels(self, labels):
        labels = tf.sparse.to_dense(labels)
        labels = tf.py_function(self._parse_label, [labels], [tf.int8])[0]
        labels.set_shape([self.num_labels])
        return labels

    def parse(self, features, ifc=False):
        input_ids_title, mask_title = self._parse_title(features['title'], ifc)
        input_ids_asr, mask_asr = self._parse_asr(features['asr_text'], ifc)
        frames, num_frames = self._parse_frames(
            features['frame_feature'], ifc)  # fames(32, 1536) num_frames(1,) bsz=256 dim_v_fea=1536 len_v = 32
        # print('#############',frames, num_frames, '#############')
        labels = self._parse_labels(features['tag_id'])
        return {'input_ids_title': input_ids_title, 'mask_title': mask_title,
                'input_ids_asr': input_ids_asr, 'mask_asr': mask_asr,
                'frames': frames, 'num_frames': num_frames,
                'vid': features['id'], 'labels': labels}

    def create_dataset(self, files, training, batch_size):
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
        feature_map = {'id': tf.io.FixedLenFeature([], tf.string),
                       'title': tf.io.FixedLenFeature([], tf.string),
                       'asr_text': tf.io.FixedLenFeature([], tf.string),
                       'frame_feature': tf.io.VarLenFeature(tf.string),
                       'tag_id': tf.io.VarLenFeature(tf.int64)}
        dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_map), num_parallel_calls=AUTOTUNE)

        dataset_1 = dataset.map(lambda x: self.parse(x, ifc=False), num_parallel_calls=AUTOTUNE)
        dataset_1 = dataset_1.batch(batch_size, drop_remainder=training)
        dataset_1 = dataset_1.prefetch(buffer_size=AUTOTUNE)
        
        dataset_2 = dataset.map(lambda x: self.parse(x, ifc=True), num_parallel_calls=AUTOTUNE)
        dataset_2 = dataset_2.batch(batch_size, drop_remainder=training)
        dataset_2 = dataset_2.prefetch(buffer_size=AUTOTUNE)

        return dataset_1, dataset_2


def create_datasets(args):
    train_files = glob.glob(args.train_record_pattern)
    val_files = glob.glob(args.val_record_pattern)

    parser = FeatureParser(args)
    train_dataset_1, train_dataset_2 = parser.create_dataset(train_files, training=True, batch_size=args.batch_size)
    val_dataset_1, val_dataset_2 = parser.create_dataset(val_files, training=False, batch_size=args.val_batch_size)

    return train_dataset_1, val_dataset_1, train_dataset_2, val_dataset_2
