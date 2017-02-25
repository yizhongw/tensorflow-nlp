#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 17-2-24 上午11:30
import gzip
import os

import pickle
import tensorflow as tf
from util import lazy_property
from data_helper import gen_rnn_batches


class RNNModel:
    """
    Support many RNN variants, including (bi-)lstm, (bi-)gru, (bi-)rnn
    """

    def __init__(self, embeddings, num_classes, model_config=None):
        self.embeddings = embeddings
        self.num_classes = num_classes
        self.config = model_config
        # Placeholders
        self.input = tf.placeholder(tf.int32, [None, None])
        self.input_length = tf.placeholder(tf.int32, [None])
        self.target = tf.placeholder(tf.int32, [None, num_classes])
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        if self.config is not None:
            # Run all the properties once so that tensorflow can detect all variables
            self.score, self.loss, self.prediction, self.error, self.train_op
            # Initialize all variables
            self.sess.run(tf.global_variables_initializer())

    @lazy_property
    def score(self):
        with tf.device('/cpu:0'), tf.variable_scope('embedding'):
            embeddings = tf.get_variable('embeddings',
                                         shape=self.embeddings.shape,
                                         initializer=tf.constant_initializer(self.embeddings),
                                         trainable=True)
            embedded_input = tf.nn.embedding_lookup(embeddings, self.input)

        with tf.variable_scope('rnn'):
            rnn_outputs, final_h_state = self._rnn(embedded_input, self.input_length)

        with tf.variable_scope('predict'):
            W = tf.get_variable('W',
                                shape=[self.config.getint('hidden_unit_size'), self.num_classes],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('b', shape=[self.num_classes], initializer=tf.constant_initializer(0.1))
            scores = tf.nn.xw_plus_b(final_h_state, W, b)
        return scores

    @lazy_property
    def loss(self):
        losses = tf.nn.softmax_cross_entropy_with_logits(self.score, self.target)
        return tf.reduce_mean(losses)

    @lazy_property
    def prediction(self):
        prediction = tf.argmax(self.score, 1, name='predictions')
        return prediction

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 1), self.prediction)
        return tf.reduce_mean(tf.cast(mistakes, tf.float32), name='error')

    @lazy_property
    def train_op(self):
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        if self.config.getint('max_grad_norm') is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.config.getint('max_grad_norm'))
        optimizer = tf.train.AdagradOptimizer(0.1)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        return train_op

    def train(self, train_samples, eval_func=None):
        for epoch in range(self.config.getint('train_epochs')):
            ave_loss, ave_error = 0, 0
            for batch_num, one_batch in enumerate(
                    gen_rnn_batches(train_samples, batch_size=self.config.getint('batch_size'))):
                batched_text, batched_lengths, batched_labels = zip(*one_batch)
                _, loss, error = self.sess.run([self.train_op, self.loss, self.error],
                                               feed_dict={self.input: batched_text,
                                                          self.input_length: batched_lengths,
                                                          self.target: batched_labels})
                ave_loss += loss
                ave_error += error
                if (batch_num + 1) % self.config.getint('print_per_batch_num') == 0:
                    ave_loss /= self.config.getint('print_per_batch_num')
                    ave_error /= self.config.getint('print_per_batch_num')
                    print('Epoch {:2d}, batch {:4d}: loss {:05.4f}, error {:05.4f}.'.format(epoch + 1, batch_num + 1,
                                                                                            ave_loss, ave_error))
                    ave_loss, ave_error = 0, 0
                if eval_func is not None and (batch_num + 1) % self.config.getint('eval_per_batch_num') == 0:
                    eval_func()

    def eval(self, eval_samples):
        print('Evaluation on {} samples...'.format(len(eval_samples)))
        total_error = 0
        for sample in eval_samples:
            error = self.sess.run(self.error,
                                  feed_dict={self.input: [sample[0]],
                                             self.input_length: [sample[1]],
                                             self.target: [sample[2]]})
            total_error += error
        print('Evaluation error rate {:05.4f}.'.format(total_error / len(eval_samples)))

    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        with gzip.open(os.path.join(model_dir, 'model_config.gz'), 'wb') as fout:
            pickle.dump(self.config, fout)
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(model_dir, 'rnn_checkpoint'))
        tf.reset_default_graph()
        print('Model saved.')

    def load(self, model_dir):
        with gzip.open(os.path.join(model_dir, 'model_config.gz'), 'rb') as fin:
            self.config = pickle.load(fin)
        # Run all the properties once so that tensorflow can detect all variables
        self.score, self.loss, self.prediction, self.error, self.train_op
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(model_dir, 'rnn_checkpoint'))
        print('Model loaded!')

    def _get_cell(self):
        if self.config.get('rnn_type').endswith('lstm'):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.getint('hidden_unit_size'), state_is_tuple=True)
        if self.config.get('rnn_type').endswith('gru'):
            cell = tf.nn.rnn_cell.GRUCell(num_units=self.config.getint('hidden_unit_size'))
        if self.config.get('rnn_type').endswith('rnn'):
            cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.config.getint('hidden_unit_size'))
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             self.config.getfloat('input_keep_prob'),
                                             self.config.getfloat('output_keep_prob'))
        return cell

    def _rnn(self, embed, length):
        if not self.config.get('rnn_type').startswith('bi'):
            cell = self._get_cell()
            outputs, state = tf.nn.dynamic_rnn(cell, inputs=embed, sequence_length=length, dtype=tf.float32)
            if self.config.get('rnn_type').endswith('lstm'):
                c, h = state
                state = h
        else:
            cell_fw = self._get_cell()
            cell_bw = self._get_cell()
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_bw, cell_fw, inputs=embed,
                                                             sequence_length=tf.to_int64(length), dtype=tf.float32)
            outputs = outputs[0] + outputs[1]
            state_fw, state_bw = state
            if self.config.get('rnn_type').endswith('lstm'):
                c_fw, h_fw = state_fw
                c_bw, h_bw = state_bw
                state_fw, state_bw = h_fw, h_bw
            state = state_fw + state_bw
        return outputs, state
