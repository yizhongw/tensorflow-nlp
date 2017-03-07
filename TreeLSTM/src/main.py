#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 17-3-6 下午9:39

import codecs
import os

from nltk.tokenize import sexpr
import numpy as np
from six.moves import urllib
import tensorflow as tf
import tensorflow_fold as td

from config import PathConfig, ModelConfig
from model import BinaryTreeLSTMCell, TreeLSTMModel
from util import filter_glove, load_trees, load_embeddings





def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.prepare:
        filter_glove()

    if args.train or args.test:
        weight_matrix, word_idx = load_embeddings(PathConfig.filtered_glove_path)

    if args.train:
        train_trees = load_trees(PathConfig.train_path)
        dev_trees = load_trees(PathConfig.dev_path)

    if args.test:
        test_trees = load_trees(PathConfig.test_path)

    if args.train:
        tree_lstm_model = TreeLSTMModel(weight_matrix, word_idx, ModelConfig)

        train_set = tree_lstm_model.compiler.build_loom_inputs(train_trees)

        dev_feed_dict = tree_lstm_model.compiler.build_feed_dict(dev_trees)

        best_accuracy = 0.0
        save_path = os.path.join(PathConfig.data_path, 'sentiment_model')
        for epoch, shuffled in enumerate(td.epochs(train_set, ModelConfig.epochs), 1):
            train_loss = tree_lstm_model.train_epoch(shuffled, ModelConfig.batch_size)
            accuracy = tree_lstm_model.dev_eval(epoch, train_loss, dev_feed_dict)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                checkpoint_path = tree_lstm_model.saver.save(tree_lstm_model.sess, save_path, global_step=epoch)
                print('model saved in file: %s' % checkpoint_path)

    if args.test:
        checkpoint_path = '/home/yizhong/Workspace/DeepLeaning/tensorflow-implementations/data/sentiment_model-18'
        tree_lstm_model = TreeLSTMModel(weight_matrix, word_idx, ModelConfig)
        tree_lstm_model.saver.restore(tree_lstm_model.sess, checkpoint_path)

        test_results = sorted(tree_lstm_model.sess.run(tree_lstm_model.metrics,
                                                       tree_lstm_model.compiler.build_feed_dict(test_trees)).items())
        print('    loss: [%s]' % ' '.join(
            '%s: %.3e' % (name.rsplit('_', 1)[0], v)
            for name, v in test_results if name.endswith('_loss')))
        print('accuracy: [%s]' % ' '.join(
            '%s: %.2f' % (name.rsplit('_', 1)[0], v * 100)
            for name, v in test_results if name.endswith('_hits')))
