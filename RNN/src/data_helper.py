#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 17-2-25 上午10:45

import random
import pytreebank
import numpy as np


def load_embeddings(embedding_file):
    print('Load embeddings...')
    vocab, embeddings = {}, []
    vocab['<UNK>'] = 0
    with open(embedding_file, 'r') as fin:
        for line in fin:
            try:
                line_info = line.strip().split()
                word = line_info[0]
                embedding = [float(val) for val in line_info[1:]]
                vocab[word] = len(vocab)
                embeddings.append(embedding)
            except:
                pass
    embeddings = np.array(embeddings)
    embeddings = np.concatenate(([[0] * embeddings.shape[1]], embeddings))
    print('Vocabulary size: {}'.format(len(vocab)))
    return vocab, embeddings


def load_data(data_file):
    print('Load data from {}...'.format(data_file))
    data = pytreebank.import_tree_corpus(data_file)
    data = [sample for doc in data for sample in doc.to_labeled_lines()]
    return data


def gen_rnn_batches(samples, batch_size=1, shuffle=True):
    if shuffle:
        random.shuffle(samples)
    data_size = len(samples)
    num_batches = int((data_size - 1) / batch_size) + 1
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield pad_with_zero(samples[start_index: end_index])


def pad_with_zero(samples):
    max_len = max([sample[1] for sample in samples])
    padded = []
    for sample in samples:
        texts = [id for id in sample[0]] + [0] * (max_len - len(sample[0]))
        padded.append((texts, sample[1], sample[2]))
    return padded


def convert_to_numeric_samples(raw_samples, vocab, num_classes):
    numeric_samples = []
    for sample in raw_samples:
        label = int(sample[0])
        one_hot_label = [0] * num_classes
        one_hot_label[label - 1] = 1
        words = sample[1].split()
        numeric_samples.append([transform_words(words, vocab), len(words), one_hot_label])
    return numeric_samples


def transform_words(words, vocab, padding_length=None):
    word_ids = []
    for word in words:
        if word in vocab:
            word_ids.append(vocab[word])
        else:
            word_ids.append(0)
    if padding_length is not None:
        if len(word_ids) < padding_length:
            word_ids += [0] * (padding_length - len(words))
        elif len(word_ids) > padding_length:
            word_ids = word_ids[:padding_length]
    return word_ids
