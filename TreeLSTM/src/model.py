#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 17-2-26 下午3:43
import tensorflow as tf
import tensorflow_fold as td
from util import tokenize


class BinaryTreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    """LSTM with two state inputs.

    This is the model described in section 3.2 of 'Improved Semantic
    Representations From Tree-Structured Long Short-Term Memory
    Networks' <http://arxiv.org/pdf/1503.00075.pdf>, with recurrent
    dropout as described in 'Recurrent Dropout without Memory Loss'
    <http://arxiv.org/pdf/1603.05118.pdf>.
    """

    def __init__(self, num_units, keep_prob=1.0):
        """Initialize the cell.

        Args:
            num_units: int, The number of units in the LSTM cell.
            keep_prob: Keep probability for recurrent dropout.
        """
        super(BinaryTreeLSTMCell, self).__init__(num_units)
        self._keep_prob = keep_prob

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            lhs, rhs = state
            c0, h0 = lhs
            c1, h1 = rhs
            concat = tf.contrib.layers.linear(
                tf.concat([inputs, h0, h1], 1), 5 * self._num_units)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f0, f1, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            j = self._activation(j)
            if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
                j = tf.nn.dropout(j, self._keep_prob)

            new_c = (c0 * tf.sigmoid(f0 + self._forget_bias) +
                     c1 * tf.sigmoid(f1 + self._forget_bias) +
                     tf.sigmoid(i) * j)
            new_h = self._activation(new_c) * tf.sigmoid(o)

            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


class TreeLSTMModel(object):
    def __init__(self, weight_matrix, word_idx, ModelConfig):

        self.ModelConfig = ModelConfig

        self.word_embedding = td.Embedding(*weight_matrix.shape, initializer=weight_matrix, name='word_embedding')
        self.word_idx = word_idx

        self.keep_prob_ph = tf.placeholder_with_default(1.0, [])
        self.tree_lstm = td.ScopedLayer(
            tf.contrib.rnn.DropoutWrapper(
                BinaryTreeLSTMCell(self.ModelConfig.lstm_num_units, keep_prob=self.keep_prob_ph),
                input_keep_prob=self.keep_prob_ph, output_keep_prob=self.keep_prob_ph),
            name_or_scope='tree_lstm')
        self.output_layer = td.FC(self.ModelConfig.num_classes, activation=None, name='output_layer')

        self.embed_subtree = td.ForwardDeclaration(name='embed_subtree')
        self.model = self.embed_tree(is_root=True)
        self.embed_subtree.resolve_to(self.embed_tree(is_root=False))

        self.compiler = td.Compiler.create(self.model)
        print('input type: %s' % self.model.input_type)
        print('output type: %s' % self.model.output_type)

        self.metrics = {k: tf.reduce_mean(v) for k, v in self.compiler.metric_tensors.items()}

        self.loss = tf.reduce_sum(self.compiler.metric_tensors['all_loss'])
        opt = tf.train.AdagradOptimizer(ModelConfig.learning_rate)

        grads_and_vars = opt.compute_gradients(self.loss)
        found = 0
        for i, (grad, var) in enumerate(grads_and_vars):
            if var == self.word_embedding.weights:
                found += 1
                grad = tf.scalar_mul(ModelConfig.embedding_learning_rate, grad)
                grads_and_vars[i] = (grad, var)
        assert found == 1  # internal consistency check
        self.train_op = opt.apply_gradients(grads_and_vars)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train_epoch(self, train_set, batch_size):
        loss = 0
        for batch in td.group_by_batches(train_set, batch_size):
            train_feed_dict = {self.keep_prob_ph: self.ModelConfig.keep_prob, self.compiler.loom_input_tensor: batch}
            loss += self.train_step(train_feed_dict)
        return loss

    def train_step(self, train_feed_dict):
        _, batch_loss = self.sess.run([self.train_op, self.loss], train_feed_dict)
        return batch_loss

    def dev_eval(self, epoch, train_loss, dev_feed_dict):
        dev_metrics = self.sess.run(self.metrics, dev_feed_dict)
        dev_loss = dev_metrics['all_loss']
        dev_accuracy = ['%s: %.2f' % (k, v * 100) for k, v in
                        sorted(dev_metrics.items()) if k.endswith('hits')]
        print('epoch:%4d, train_loss: %.3e, dev_loss_avg: %.3e, dev_accuracy:\n  [%s]'
              % (epoch, train_loss, dev_loss, ' '.join(dev_accuracy)))
        return dev_metrics['root_hits']

    def embed_tree(self, is_root):
        """Creates a block that embeds trees; output is tree LSTM state."""
        return td.InputTransform(tokenize) >> td.OneOf(
            key_fn=lambda pair: pair[0] == '2',  # label 2 means neutral
            case_blocks=(self.add_metrics(is_root, is_neutral=False),
                         self.add_metrics(is_root, is_neutral=True)),
            pre_block=(td.Scalar('int32'), self.logits_and_state()))

    def logits_and_state(self):
        """Creates a block that goes from tokens to (logits, state) tuples."""
        unknown_idx = len(self.word_idx)
        lookup_word = lambda word: self.word_idx.get(word, unknown_idx)

        word2vec = (td.GetItem(0) >> td.InputTransform(lookup_word) >>
                    td.Scalar('int32') >> self.word_embedding)

        pair2vec = (self.embed_subtree(), self.embed_subtree())

        # Trees are binary, so the tree layer takes two states as its input_state.
        zero_state = td.Zeros((self.tree_lstm.state_size,) * 2)
        # Input is a word vector.
        zero_inp = td.Zeros(self.word_embedding.output_type.shape[0])

        word_case = td.AllOf(word2vec, zero_state)
        pair_case = td.AllOf(zero_inp, pair2vec)

        tree2vec = td.OneOf(len, [(1, word_case), (2, pair_case)])

        return tree2vec >> self.tree_lstm >> (self.output_layer, td.Identity())

    def add_metrics(self, is_root, is_neutral):
        """A block that adds metrics for loss and hits; output is the LSTM state."""
        c = td.Composition(
            name='predict(is_root=%s, is_neutral=%s)' % (is_root, is_neutral))
        with c.scope():
            # destructure the input; (labels, (logits, state))
            labels = c.input[0]
            logits = td.GetItem(0).reads(c.input[1])
            state = td.GetItem(1).reads(c.input[1])

            # calculate loss
            loss = td.Function(self.tf_node_loss)
            td.Metric('all_loss').reads(loss.reads(logits, labels))
            if is_root:
                td.Metric('root_loss').reads(loss)

            # calculate fine-grained hits
            hits = td.Function(self.tf_fine_grained_hits)
            td.Metric('all_hits').reads(hits.reads(logits, labels))
            if is_root:
                td.Metric('root_hits').reads(hits)

            # calculate binary hits, if the label is not neutral
            if not is_neutral:
                binary_hits = td.Function(self.tf_binary_hits).reads(logits, labels)
                td.Metric('all_binary_hits').reads(binary_hits)
                if is_root:
                    td.Metric('root_binary_hits').reads(binary_hits)

            # output the state, which will be read by our by parent's LSTM cell
            c.output.reads(state)
        return c

    @staticmethod
    def tf_node_loss(logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    @staticmethod
    def tf_fine_grained_hits(logits, labels):
        predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
        return tf.cast(tf.equal(predictions, labels), tf.float64)

    @staticmethod
    def tf_binary_hits(logits, labels):
        softmax = tf.nn.softmax(logits)
        binary_predictions = (softmax[:, 3] + softmax[:, 4]) > (softmax[:, 0] + softmax[:, 1])
        binary_labels = labels > 2
        return tf.cast(tf.equal(binary_predictions, binary_labels), tf.float64)

