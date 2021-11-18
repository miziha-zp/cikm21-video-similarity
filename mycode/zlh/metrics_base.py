import logging
import tensorflow as tf


class Recorder:
    def __init__(self):
        self.loss = tf.keras.metrics.Mean()
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        self.loss_1 = tf.keras.metrics.Mean()
        self.loss_2 = tf.keras.metrics.Mean()
        self.loss_3 = tf.keras.metrics.Mean()
        self.pattern = 'Epoch: {}, step: {}, loss: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, loss_1: {:.4f}, loss_2: {:.4f}, loss_3: {:.4f}'

    def record(self, losses, labels, predictions, loss_1, loss_2, loss_3):
        self.loss.update_state(losses)
        self.loss_1.update_state(loss_1)
        self.loss_2.update_state(loss_2)
        self.loss_3.update_state(loss_3)
        self.precision.update_state(labels, predictions)
        self.recall.update_state(labels, predictions)

    def reset(self):
        self.loss.reset_states()
        self.loss_1.reset_states()
        self.loss_2.reset_states()
        self.loss_3.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()

    def _results(self):
        loss = self.loss.result().numpy()
        loss_1 = self.loss_1.result().numpy()
        loss_2 = self.loss_2.result().numpy()
        loss_3 = self.loss_3.result().numpy()
        precision = self.precision.result().numpy()
        recall = self.recall.result().numpy()
        f1 = 2 * precision * recall / (precision + recall + 1e-6)  # avoid division by 0
        return [loss, precision, recall, f1, loss_1, loss_2, loss_3]

    def score(self):
        return self._results()[-1].numpy()

    def log(self, epoch, num_step, prefix='', suffix=''):
        loss, precision, recall, f1, loss_1, loss_2, loss_3 = self._results()
        logging.info(prefix + self.pattern.format(epoch, num_step, loss, precision, recall, f1, loss_1, loss_2, loss_3) + suffix)
