"""
Seq2Seq model which maps
(initial state, command) -> sequence of actions
"""
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
import numpy as np
from collections import namedtuple

from semantirl.data import load_turk_train_limited
from semantirl.utils import Vocab, PAD, EOS, pad, one_hot

MAX_LEN = 20

DataPoint = namedtuple('DataPoint', ['obs', 'acts', 'sentence'])


def assert_shape(tensor, shape):
    assert tensor.get_shape().is_compatible_with(shape),\
        'Incompatible shapes: %s vs %s' % (tensor.get_shape(), shape)

def load_data():
    trajs = [traj for traj in load_turk_train_limited()]

    # Prune trajectories with long sentences (only about 20)
    trajs = [traj for traj in trajs if len(traj.sentence) <= MAX_LEN]
    sents = [traj.sentence+[EOS] for traj in trajs]

    # Calculate max len & pad sentences
    max_len = 0
    for sent in sents:
        max_len = max(max_len, len(sent))
    sents = [pad(sent, PAD, max_len) for sent in sents]
    
    # Build vocab
    vocab = Vocab()
    for sent in sents:
        for word in sent:
            vocab.add(word)
    vocab = vocab.prune_rares(cutoff=2)

    # Pad actions
    acts = [traj.actions for traj in trajs]
    max_len = np.max([len(act) for act in acts])
    acts = [pad(act, PAD, max_len) for act in acts]
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

    return vocab, avocab, dataset, max_len, init_state.shape


def build_model(obs_shape, vocab, avocab, 
        max_cmd=MAX_LEN, max_act=MAX_LEN, batch_size=1):
    dim_hidden = 10
    embed_size = 10
    num_actions = len(avocab)
    obs_shape = list(obs_shape)

    obs = tf.placeholder(tf.float32, [batch_size]+obs_shape)
    sentence = tf.placeholder(tf.int32, [batch_size, max_cmd])
    actions = tf.placeholder(tf.int32, [batch_size, max_act])

    with tf.variable_scope('cmd2seq'):
        embedding_matrix = tf.get_variable('WEmbed', (len(vocab), embed_size))
        word_embeddings = tf.nn.embedding_lookup(embedding_matrix, sentence)
        assert_shape(word_embeddings, (batch_size, max_cmd, embed_size))

        action_embedding_mat = tf.get_variable('AEmbed', (num_actions, embed_size))
        action_embeddings = tf.nn.embedding_lookup(action_embedding_mat, actions)
        assert_shape(action_embeddings, (batch_size, max_act, embed_size))

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

        # Project outputs
        Wact = tf.get_variable('Wact', (dim_hidden, num_actions))
        bact = tf.get_variable('bact', (num_actions))
        act_output = tf.matmul(tf.reshape(dec_outputs, [-1, dim_hidden]), Wact)+bact
        act_output = tf.reshape(act_output, [max_act, batch_size, num_actions])
        act_output = tf.transpose(act_output, [1,0,2])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(act_output, actions)
        batch_loss = tf.reduce_sum(loss)


def main():
    vocab, avocab, dataset, max_alen, obs_shape = load_data()

    build_model(obs_shape, vocab, avocab, max_act=max_alen)

if __name__ == "__main__":
    main()

