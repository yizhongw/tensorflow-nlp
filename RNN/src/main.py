#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 17-2-24 上午11:30

import os
import argparse
import configparser
import data_helper
from model import RNNModel

config = configparser.ConfigParser()
config.read('config.ini')


def train_model():
    vocab, embeddings = data_helper.load_embeddings(config.get('data', 'embedding_file'))
    train_data = data_helper.load_data(os.path.join(config.get('data', 'treebank_dir'), 'train.txt'))
    numeric_train_samples = data_helper.convert_to_numeric_samples(train_data, vocab, num_classes=5)
    model = RNNModel(embeddings, num_classes=5, model_config=config['model'])
    dev_data = data_helper.load_data(os.path.join(config.get('data', 'treebank_dir'), 'dev.txt'))
    numeric_dev_samples = data_helper.convert_to_numeric_samples(dev_data, vocab, num_classes=5)
    eval_func = lambda: model.eval(numeric_dev_samples)
    model.train(numeric_train_samples, eval_func)
    model.save(config.get('data', 'model_dir'))


def test_model():
    vocab, embeddings = data_helper.load_embeddings(config.get('data', 'embedding_file'))
    model = RNNModel(embeddings, num_classes=5)
    model.load(config.get('data', 'model_dir'))
    test_data = data_helper.load_data(os.path.join(config.get('data', 'treebank_dir'), 'test.txt'))
    numeric_test_samples = data_helper.convert_to_numeric_samples(test_data, vocab, num_classes=5)
    model.eval(numeric_test_samples)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.train:
        train_model()
    if args.test:
        test_model()