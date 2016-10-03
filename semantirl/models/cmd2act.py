"""
Seq2Seq model which maps
(initial state, command) -> sequence of actions
"""
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
import numpy as np
from collections import namedtuple

from semantirl.data import load_turk_train_limited
from semantirl.utils import Vocab, PAD, EOS, pad, one_hot, BatchSampler

MAX_LEN = 20
START = '_START'

DataPoint = namedtuple('DataPoint', ['obs', 'acts', 'sentence'])


def assert_shape(tensor, shape):
    assert tensor.get_shape().is_compatible_with(shape),\
        'Incompatible shapes: %s vs %s' % (tensor.get_shape(), shape)

def load_data():
    trajs = [traj for traj in load_turk_train_limited()]

    # Prune trajectories with long sentences (only about 20)
    trajs = [traj for traj in trajs if len(traj.sentence) <= (MAX_LEN)]
    sents = [traj.sentence+[EOS] for traj in trajs]

    # Calculate max len & pad sentences
    max_len = 0
    for sent in sents:
        max_len = max(max_len, len(sent))
    sents = [pad(sent, PAD, max_len) for sent in sents]
    max_sent = max_len
    
    # Build vocab
    vocab = Vocab()
    for sent in sents:
        for word in sent:
            vocab.add(word)
    vocab = vocab.prune_rares(cutoff=2)

    # Pad actions
    acts = [traj.actions for traj in trajs]
    max_len = np.max([len(act) for act in acts])
    acts = [[START]+pad(act, PAD, max_len) for act in acts]
    max_len += 1
    avocab = Vocab()
    for act_seq in acts:
        for act in act_seq:
            avocab.add(act)

    # Construct data
    dataset = []
    for i in range(len(trajs)):
        init_state = np.reshape(trajs[i].states[0].to_dense(), [-1])  # TODO: Flatten hack
        dataset.append(DataPoint(init_state,
                                 avocab.words2indices(acts[i]),
                                 vocab.words2indices(sents[i])))

    return vocab, avocab, dataset, max_sent, max_len, init_state.shape


class Cmd2Act(object):
    def __init__(self, batch_size=1):
        self.batch_size = batch_size

    def _build_model(self, obs_shape, vocab, avocab, 
            max_cmd=MAX_LEN, max_act=MAX_LEN):
        dim_hidden = 10
        embed_size = 10
        num_actions = len(avocab)
        obs_shape = list(obs_shape)
        batch_size = self.batch_size

        self.obs = obs = tf.placeholder(tf.float32, [batch_size]+obs_shape)
        self.sentence = sentence = tf.placeholder(tf.int32, [batch_size, max_cmd])
        self.actions = actions = tf.placeholder(tf.int32, [batch_size, max_act])
        action_labels = actions[:,1:]
        assert_shape(action_labels, (batch_size, max_act-1))
        actions = actions[:,:-1]
        self.lr = tf.placeholder(tf.float32, [])

        with tf.variable_scope('cmd2seq'):
            embedding_matrix = tf.get_variable('WEmbed', (len(vocab), embed_size))
            word_embeddings = tf.nn.embedding_lookup(embedding_matrix, sentence)
            assert_shape(word_embeddings, (batch_size, max_cmd, embed_size))

            action_embedding_mat = tf.get_variable('AEmbed', (num_actions, embed_size))
            action_embeddings = tf.nn.embedding_lookup(action_embedding_mat, actions)
            assert_shape(action_embeddings, (batch_size, max_act-1, embed_size))

            # Project obs into dim_hidden (TODO: Use convnet)
            Wobs = tf.get_variable('Wobs', obs_shape+[dim_hidden])
            bobs = tf.get_variable('bobs', (dim_hidden,))
            obs_proj = tf.matmul(obs, Wobs)+bobs
            assert_shape(obs_proj, (batch_size, dim_hidden))

            # Embed sentence via RNN
            with tf.variable_scope('encoder'):
                enc_cell = rnn_cell.BasicRNNCell(dim_hidden)
                # unpack tensors
                word_list = tf.unpack(word_embeddings, axis=1)
                enc_outputs, last_state = rnn.rnn(enc_cell, word_list, initial_state=obs_proj)
                assert_shape(last_state, (batch_size, dim_hidden))

            # Decode actions via RNN
            with tf.variable_scope('decoder'):
                dec_cell = rnn_cell.BasicRNNCell(dim_hidden)
                action_list = tf.unpack(action_embeddings, axis=1)
                #TODO: Offset actions by 1
                dec_outputs, dec_state = rnn.rnn(dec_cell, action_list, initial_state=last_state)

            # Project decoder outputs into actions
            Wact = tf.get_variable('Wact', (dim_hidden, num_actions))
            bact = tf.get_variable('bact', (num_actions))
            act_output = tf.matmul(tf.reshape(dec_outputs, [-1, dim_hidden]), Wact)+bact
            self.act_logits = tf.nn.softmax(act_output)
            act_output = tf.reshape(act_output, [max_act-1, batch_size, num_actions])
            act_output = tf.transpose(act_output, [1,0,2])

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(act_output, action_labels)
            batch_loss = tf.reduce_mean(loss)

        self.batch_loss = batch_loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(batch_loss)

    def _init_tf(self):
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def run(self, fetches, feeds={}):
        return self.sess.run(fetches, feed_dict=feeds)

    def _make_feed(self, batch):
        obs_batch = np.r_[[datum[0] for datum in batch]]
        act_batch = np.r_[[datum[1] for datum in batch]]
        sent_batch = np.r_[[datum[2] for datum in batch]]
        return {self.obs: obs_batch, self.sentence:sent_batch, self.actions:act_batch}

    def train_step(self, batch):
        """
        Run one step of gradient descent

        Args:
            batch: A list of DataPoint objects

        Returns:
            loss (float)
        """
        feeds = self._make_feed(batch)
        feeds[self.lr] = 1e-2
        loss, _ = self.run([self.batch_loss, self.train_op], feeds=feeds)
        return loss

    def train(self, dataset, heartbeat=1000):
        sampler = BatchSampler(dataset)
        for i, batch in enumerate(sampler.with_replacement(batch_size=self.batch_size)):
            if i%heartbeat == 0:
                print i, self.train_step(batch)

                feeds = self._make_feed(batch)
                #print self.run(self.act_logits, feeds=feeds)


def main():
    np.set_printoptions(suppress=True)
    vocab, avocab, dataset, max_slen, max_alen, obs_shape = load_data()

    cmd = Cmd2Act(batch_size=5)
    cmd._build_model(obs_shape, vocab, avocab, max_cmd=max_slen, max_act=max_alen)
    cmd._init_tf()
    cmd.train(dataset)


if __name__ == "__main__":
    main()

