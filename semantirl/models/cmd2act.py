"""
Seq2Seq model which maps
(initial state, command) -> sequence of actions
"""
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
import numpy as np
from collections import namedtuple

from semantirl.data import load_turk_train_limited
from semantirl.utils import Vocab, PAD, EOS, START, pad, split_train_test, BatchSampler, pretty_print_sentence
from semantirl.utils.tf_utils import get_wt_string, restore_wt_string, dump_pickle_str, load_pickle_str

MAX_LEN = 20

DataPoint = namedtuple('DataPoint', ['obs', 'acts', 'sentence', 'traj'])


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
    acts = [[START]+act+[EOS] for act in acts]
    max_alen = np.max([len(act) for act in acts])
    acts = [pad(act, PAD, max_alen) for act in acts]
    avocab = Vocab()
    for act_seq in acts:
        for act in act_seq:
            avocab.add(act)

    # Construct data
    dataset = []
    for i in range(len(trajs)):
        init_state = trajs[i].states[0].to_dense()
        dataset.append(DataPoint(init_state,
                                 avocab.words2indices(acts[i]),
                                 vocab.words2indices(sents[i]),
                                 trajs[i]
                                 )
                       )

    return vocab, avocab, dataset, max_sent, max_alen, init_state.shape


def project_obs(obs, dim_out):
    """Convolve initial state """
    batch_size, w, h, ch = obs.get_shape()

    with tf.variable_scope('conv_obs'):
        filt1 = tf.get_variable('filt1', (5, 5, ch, 1))
        conv1 = tf.nn.conv2d(obs, filt1, [1,2,2,1], 'VALID')
        _, w, h, ch = conv1.get_shape()
        flat_dim = int(w*h*ch)
        flattened = tf.reshape(conv1, [-1, flat_dim])
        Wfull = tf.get_variable('Wfull', (flat_dim, dim_out))
        bfull = tf.get_variable('bfull', (dim_out))
        flat_out = tf.matmul(flattened, Wfull)+bfull
    return flat_out


class Cmd2Act(object):
    def __init__(self, cmd_vocab, act_vocab, obs_shape, max_cmd_len=MAX_LEN, max_act_len=MAX_LEN):
        self.cmd_vocab = cmd_vocab
        self.act_vocab = act_vocab
        self.max_cmd = max_cmd_len
        self.max_act = max_act_len
        self.obs_shape = obs_shape
        self._build_model()
        self._init_tf()

    def _build_model(self):
        dim_hidden = 10
        embed_size = 10
        num_actions = len(self.act_vocab)
        cmd_vocab_size = len(self.cmd_vocab)
        obs_shape = list(self.obs_shape)
        max_act = self.max_act
        max_cmd = self.max_cmd

        self.obs = obs = tf.placeholder(tf.float32, [None]+obs_shape)
        obs = tf.transpose(obs, [0, 2, 3, 1])

        self.sentence = sentence = tf.placeholder(tf.int32, [None, max_cmd])
        self.actions = actions = tf.placeholder(tf.int32, [None, max_act])
        self.action_labels = action_labels = actions[:,1:]  # First action is START placeholder
        assert_shape(action_labels, (None, max_act-1))
        actions = actions[:,:-1]
        self.lr = tf.placeholder(tf.float32, [])
        self.dec_input = tf.placeholder(tf.int32, [None,])
        self.dec_state = tf.placeholder(tf.float32, [None, dim_hidden])

        with tf.variable_scope('cmd2seq'):
            embedding_matrix = tf.get_variable('WEmbed', #(cmd_vocab_size, embed_size),
                                                initializer=(2*np.random.rand(cmd_vocab_size,embed_size)-1.).astype(np.float32))
            word_embeddings = tf.nn.embedding_lookup(embedding_matrix, sentence)
            assert_shape(word_embeddings, (None, max_cmd, embed_size))

            action_embedding_mat = tf.get_variable('AEmbed', (num_actions, dim_hidden))
            action_embeddings = tf.nn.embedding_lookup(action_embedding_mat, actions)
            assert_shape(action_embeddings, (None, max_act-1, dim_hidden))

            dec_action_embed = tf.nn.embedding_lookup(action_embedding_mat, self.dec_input)
            assert_shape(dec_action_embed, (None, dim_hidden))


            # Project obs into dim_hidden (TODO: Use convnet)
            obs_proj = project_obs(obs, dim_hidden)
            #Wobs = tf.get_variable('Wobs', obs_shape+[dim_hidden])
            #bobs = tf.get_variable('bobs', (dim_hidden,))
            #obs_proj = tf.matmul(obs, Wobs)+bobs
            assert_shape(obs_proj, (None, dim_hidden))
            self.obs_proj = obs_proj

            # Embed sentence via RNN
            with tf.variable_scope('encoder'):
                enc_cell = rnn_cell.GRUCell(dim_hidden)
                # unpack tensors
                word_list = tf.unpack(word_embeddings, axis=1)
                enc_outputs, last_state = rnn.rnn(enc_cell, word_list, initial_state=obs_proj)
                assert_shape(last_state, (None, dim_hidden))

                self.encoding = last_state

            # Decode actions via RNN
            with tf.variable_scope('decoder'):
                dec_cell = rnn_cell.GRUCell(dim_hidden)
                action_list = tf.unpack(action_embeddings, axis=1)
                self.dec_action_embed = action_list

                with tf.variable_scope('decoder') as varscope:
                    dec_outputs, dec_state = rnn.rnn(dec_cell, action_list, initial_state=last_state, scope=varscope)
                self.dec_outputs = dec_outputs
                self.dec_state = dec_state

                # Test-time decoding, one step
                with tf.variable_scope('decoder', reuse=True):
                    output, next_state = dec_cell(dec_action_embed, self.dec_state)
                self.step_act_input = dec_action_embed
                self.step_rnn_out = output
                self.step_state = next_state

            # Project decoder outputs into actions
            Wact = tf.get_variable('Wact', (dim_hidden, num_actions))
            bact = tf.get_variable('bact', (num_actions))
            act_output = tf.transpose(dec_outputs, [1,0,2])
            act_output = tf.reshape(act_output, [-1, dim_hidden])
            act_output = tf.matmul(act_output, Wact)+bact
            self.act_logits = tf.nn.softmax(act_output)
            self.act_logits = tf.reshape(self.act_logits, [-1, max_act-1, num_actions])
            act_output = tf.reshape(act_output, [-1, max_act-1, num_actions])

            step_action_out = tf.reshape(self.step_rnn_out, [-1, dim_hidden])
            step_action_out = tf.matmul(step_action_out, Wact)+bact
            self.step_action_out = tf.nn.softmax(step_action_out)

            self.act_max = tf.argmax(self.act_logits, 2)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(act_output, action_labels)
            batch_loss = tf.reduce_mean(loss)

        self.batch_loss = batch_loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(batch_loss)

    def _init_tf(self):
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def run(self, fetches, feeds=None):
        return self.sess.run(fetches, feed_dict=feeds)

    def _make_feed(self, batch):
        obs_batch = np.r_[[datum.obs for datum in batch]]
        act_batch = np.r_[[datum.acts for datum in batch]]
        sent_batch = np.r_[[datum.sentence for datum in batch]]
        return {self.obs: obs_batch, self.sentence:sent_batch, self.actions:act_batch}

    def encode_input(self, init_state, cmd):
        init_state = np.expand_dims(init_state, 0)
        cmd_idx = self.cmd_vocab.words2indices(pad(cmd+[EOS], PAD, self.max_cmd))
        cmd_idx = np.expand_dims(cmd_idx, 0)
        feed_dict = {self.obs: init_state, self.sentence: cmd_idx}
        encoding, obs_proj = self.run([self.encoding, self.obs_proj], feeds=feed_dict)
        return encoding[0], obs_proj[0]

    def decode_step(self, encoding, input):
        input = self.act_vocab.words2indices([input])[0]
        input = np.expand_dims(input, 0)
        encoding = np.expand_dims(encoding, 0)
        feed_dict = {self.dec_input:input, self.dec_state:encoding}
        step_action, step_rnn, step_state, step_act_enc = \
                self.run([self.step_action_out, self.step_rnn_out, self.step_state, self.step_act_input], feeds=feed_dict)
        argmax_output = np.argmax(step_action)
        output_action = self.act_vocab.indices2words([argmax_output])[0]
        return output_action, step_state[0]

    def decode_full(self, encoding):
        next_state = encoding
        out_action = START
        all_acts = []
        for i in range(self.max_act):
            out_action, next_state = self.decode_step(next_state, out_action)
            all_acts.append(out_action)
        return all_acts

    def debug(self, ex):
        ex = [ex]
        feeds = self._make_feed(ex)
        actions, action_labels, enc, dec, dec_act, dec_lstate = \
            self.run([self.actions, self.action_labels, self.encoding,
                     self.dec_outputs, self.dec_action_embed,
                     self.dec_state], feeds= feeds)
        dec = np.array(dec)[:,0,:]
        dec_act = np.array(dec_act)[:,0,:]

        print '--DEBUG--'
        print self.act_vocab.indices2words(range(len(self.act_vocab)))
        print 'Actions:', actions
        print 'ActLabl:', action_labels
        print 'Encodng:', enc
        print 'DecOutp:', dec

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

    def eval_loss(self, batch):
        feeds = self._make_feed(batch)
        loss = self.run(self.batch_loss, feeds=feeds)
        return loss

    def train(self, dataset, test_dataset=None, max_iter=4000, heartbeat=100, batch_size=5):
        sampler = BatchSampler(dataset)
        for i, batch in enumerate(sampler.with_replacement(batch_size=batch_size)):
            loss = self.train_step(batch)
            if i%heartbeat == 0:
                print '=='*10, 'Iter:', i
                print 'Train loss:', loss
                if test_dataset:
                    print 'Test loss:', self.eval_loss(test_dataset)

                # Test set printouts
                for t in range(2):
                    print '------- Test Ex.', t
                    example = test_dataset[t]
                    traj = example.traj
                    feeds = self._make_feed([example])
                    results = self.run([self.act_max, self.action_labels], feeds=feeds)
                    act_max, act_labels = results
                    print 'Sentence:', ' '.join(traj.sentence)
                    #print 'Pred actions:', self.act_vocab.indices2words(act_max[0])
                    print 'GT actions:', ' '.join(pretty_print_sentence(self.act_vocab.indices2words(act_labels[0])))
                    enc, obs_proj = self.encode_input(example.obs, traj.sentence)
                    #print 'Encoding:', enc
                    dec = self.decode_full(enc)
                    print 'Test Decode:', ' '.join(pretty_print_sentence(dec))
            if i>max_iter:
                break

    def compute_action_sequence(self, init_state, sentence):
        enc, _ = self.encode_input(init_state.to_dense(), sentence)
        dec = self.decode_full(enc)
        return dec, enc

    #Pickling
    def __getstate__(self):
        return {'cmd_vocab': self.cmd_vocab,
                'act_vocab': self.act_vocab,
                'max_cmd': self.max_cmd,
                'max_act': self.max_act,
                'obs_shape': self.obs_shape,
                'wts': get_wt_string(self.saver, self.sess)
        }

    def __setstate__(self, state):
        self.__init__(state['cmd_vocab'],
                      state['act_vocab'],
                      max_cmd_len=state['max_cmd'],
                      max_act_len=state['max_act'],
                      obs_shape=state['obs_shape'])
        restore_wt_string(self.saver, self.sess, state['wts'])


def main():
    np.set_printoptions(suppress=True, precision=5)
    vocab, avocab, dataset, max_slen, max_alen, obs_shape = load_data()
    train, test = split_train_test(dataset, train_perc=0.8, shuffle=True)

    cmd = Cmd2Act(vocab, avocab, obs_shape, max_cmd_len=max_slen, max_act_len=max_alen)
    cmd.train(train, test_dataset=test)

    import pickle
    with open('cmd2act.model', 'w+b') as f:
        pickle.dump(cmd, f)


if __name__ == "__main__":
    main()

