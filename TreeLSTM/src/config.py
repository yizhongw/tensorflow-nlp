#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 17-2-26 下午3:44


class PathConfig(object):
    data_path = '/home/yizhong/Workspace/DeepLeaning/tensorflow-implementations/data/'
    sentence_path = data_path + '/stanfordSentimentTreebank/SOStr.txt'
    full_glove_path = data_path + '/embeddings/glove.6B.50d.txt'
    filtered_glove_path = data_path + '/embeddings/filtered_glove.6B.50d.txt'
    train_path = data_path + '/stanfordSentimentTreebank/trees/train.txt'
    dev_path = data_path + '/stanfordSentimentTreebank/trees/dev.txt'
    test_path = data_path + '/stanfordSentimentTreebank/trees/test.txt'


class ModelConfig(object):
    num_classes = 5
    lstm_num_units = 300  # Tai et al. used 150, but our regularization strategy is more effective
    learning_rate = 0.05
    keep_prob = 0.75
    batch_size = 100
    epochs = 20
    embedding_learning_rate = 0.1
