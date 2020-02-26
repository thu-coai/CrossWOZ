# -*- coding: utf-8 -*-

import json
import math
import os
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), 'data/mdbt')
VALIDATION_URL = os.path.join(DATA_PATH, "data/validate.json")
WORD_VECTORS_URL = os.path.join(DATA_PATH, "word-vectors/paragram_300_sl999.txt")
TRAINING_URL = os.path.join(DATA_PATH, "data/train.json")
ONTOLOGY_URL = os.path.join(DATA_PATH, "data/ontology.json")
TESTING_URL = os.path.join(DATA_PATH, "data/test.json")
MODEL_URL = os.path.join(DATA_PATH, "models/model-1")
GRAPH_URL = os.path.join(DATA_PATH, "graphs/graph-1")
RESULTS_URL = os.path.join(DATA_PATH, "results/log-1.txt")

#ROOT_URL = '../../data/mdbt'

#VALIDATION_URL = "./data/mdbt/data/validate.json"
#WORD_VECTORS_URL = "./data/mdbt/word-vectors/paragram_300_sl999.txt"
#TRAINING_URL = "./data/mdbt/data/train.json"
#ONTOLOGY_URL = "./data/mdbt/data/ontology.json"
#TESTING_URL = "./data/mdbt/data/test.json"
#MODEL_URL = "./data/mdbt/models/model-1"
#GRAPH_URL = "./data/mdbt/graphs/graph-1"
#RESULTS_URL = "./data/mdbt/results/log-1.txt"


domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']

train_batch_size = 64
batches_per_eval = 10
no_epochs = 600
device = "gpu"
start_batch = 0

num_slots = 0

booking_slots = {}

network = "lstm"
bidirect = True
lstm_num_hidden = 50
max_utterance_length = 50
vector_dimension = 300
max_no_turns = 22


# rnnrollout.py
def get_available_devs():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


class GRU(tf.nn.rnn_cell.RNNCell):
    '''
    Create a Gated Recurrent unit to unroll the network through time
    for combining the current and previous belief states
    '''

    def __init__(self, W_h, U_h, M_h, W_m, U_m, label_size, reuse=None, binary_output=False):
        super(GRU, self).__init__(_reuse=reuse)
        self.label_size = label_size
        self.M_h = M_h
        self.W_m = W_m
        self.U_m = U_m
        self.U_h = U_h
        self.W_h = W_h
        self.binary_output = binary_output

    def __call__(self, inputs, state, scope=None):
        state_only = tf.slice(state, [0, self.label_size], [-1, -1])
        output_only = tf.slice(state, [0, 0], [-1, self.label_size])
        new_state = tf.tanh(tf.matmul(inputs, self.U_m) + tf.matmul(state_only, self.W_m))
        output = tf.matmul(inputs, self.U_h) + tf.matmul(output_only, self.W_h) + tf.matmul(state_only, self.M_h)
        if self.binary_output:
            output_ = tf.sigmoid(output)
        else:
            output_ = tf.nn.softmax(output)
        state = tf.concat([output_, new_state], 1)
        return output, state

    @property
    def state_size(self):
        return tf.shape(self.W_m)[0] + self.label_size

    @property
    def output_size(self):
        return tf.shape(self.W_h)[0]


def define_CNN_model(utter, num_filters=300, name="r"):
    """
    Better code for defining the CNN model.
    """
    filter_sizes = [1, 2, 3]
    W = []
    b = []
    for i, filter_size in enumerate(filter_sizes):
        filter_shape = [filter_size, vector_dimension, 1, num_filters]
        W.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="F_W"))
        b.append(tf.Variable(tf.constant(0.1, shape=[num_filters]), name="F_b"))

    utter = tf.reshape(utter, [-1, max_utterance_length, vector_dimension])

    hidden_representation = tf.zeros([num_filters], tf.float32)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        # with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        conv = tf.nn.conv2d(
            tf.expand_dims(utter, -1),
            W[i],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv_R")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b[i]), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, max_utterance_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="r_")
        pooled_outputs.append(pooled)

        hidden_representation += tf.reshape(tf.concat(pooled, 3), [-1, num_filters])

    hidden_representation = tf.reshape(hidden_representation, [-1, max_no_turns, num_filters], name=name)

    return hidden_representation


def lstm_model(text_input, utterance_length, num_hidden, name, net_type, bidir):
    '''Define an Lstm model that will run across the user input and system act

    :param text_input: [batch_size, max_num_turns, max_utterance_size, vector_dimension]
    :param utterance_length: number words in every utterance [batch_size, max_num_turns, 1]
    :param num_hidden: -- int --
    :param name: The name of lstm network
    :param net_type: type of the network ("lstm" or "gru" or "rnn")
    :param bidir: use a bidirectional network -- bool --
    :return: output at each state [batch_size, max_num_turns, max_utterance_size, num_hidden],
     output of the final state [batch_size, max_num_turns, num_hidden]
    '''
    with tf.variable_scope(name):

        text_input = tf.reshape(text_input, [-1, max_utterance_length, vector_dimension])
        utterance_length = tf.reshape(utterance_length, [-1])

        def rnn(net_typ, num_units):
            if net_typ == "lstm":
                return tf.nn.rnn_cell.LSTMCell(num_units)
            elif net_typ == "gru":
                return tf.nn.rnn_cell.GRUCell(num_units)
            else:
                return tf.nn.rnn_cell.BasicRNNCell(num_units)

        if bidir:
            assert num_hidden % 2 == 0
            rev_cell = rnn(net_type, num_hidden // 2)
            cell = rnn(net_type, num_hidden // 2)
            _, lspd = tf.nn.bidirectional_dynamic_rnn(cell, rev_cell, text_input, dtype=tf.float32,
                                                      sequence_length=utterance_length)
            if net_type == "lstm":
                lspd = (lspd[0].h, lspd[1].h)

            last_state = tf.concat(lspd, 1)
        else:
            cell = rnn(net_type, num_hidden)
            _, last_state = tf.nn.dynamic_rnn(cell, text_input, dtype=tf.float32, sequence_length=utterance_length)
            if net_type == "lstm":
                last_state = last_state.h

        last_state = tf.reshape(last_state, [-1, max_no_turns, num_hidden])

        return last_state


def model_definition(ontology, num_slots, slots, num_hidden=None, net_type=None, bidir=None, test=False, dev=None):
    '''Create neural belief tracker model that is defined in my notes. It consists of encoding the user and system \
    input, then use the ontology to decode the encoder in manner that detects if a domain-slot-value class is mentioned
    
    :param ontology: numpy array of the embedded vectors of the ontology [num_slots, 3*vector_dimension]
    :param num_slots: number of ontology classes --int--
    :param slots: indices of the values of each slot list of lists of ints
    :param num_hidden: Number of hidden units or dimension of the hidden space
    :param net_type: The type of the encoder network cnn, lstm, gru, rnn ...etc
    :param bidir: For recurrent networks should it be bidirectional
    :param test: This is testing mode (no back-propagation)
    :param dev: Device to run the model on (cpu or gpu)
    :return: All input variable/placeholders output metrics (precision, recall, f1-score) and trainer
    '''
    # print('model definition')
    # print(ontology, num_slots, slots, num_hidden, net_type, bidir, test, dev)
    global lstm_num_hidden

    if not net_type:
        net_type = network
    else:
        print("\tMDBT: Setting up the type of the network to {}..............................".format(net_type))
    if bidir == None:
        bidir = bidirect
    else:
        pass
        # print("\tMDBT: Setting up type of the recurrent network to bidirectional {}...........................".format(bidir))
    if num_hidden:
        lstm_num_hidden = num_hidden
        print("\tMDBT: Setting up type of the dimension of the hidden space to {}.........................".format(num_hidden))

    ontology = tf.constant(ontology, dtype=tf.float32)

    # ----------------------------------- Define the input variables --------------------------------------------------
    user_input = tf.placeholder(tf.float32, [None, max_no_turns, max_utterance_length, vector_dimension], name="user")
    system_input = tf.placeholder(tf.float32, [None, max_no_turns, max_utterance_length, vector_dimension], name="sys")
    num_turns = tf.placeholder(tf.int32, [None], name="num_turns")
    user_utterance_lengths = tf.placeholder(tf.int32, [None, max_no_turns], name="user_sen_len")
    sys_utterance_lengths = tf.placeholder(tf.int32, [None, max_no_turns], name="sys_sen_len")
    labels = tf.placeholder(tf.float32, [None, max_no_turns, num_slots], name="labels")
    domain_labels = tf.placeholder(tf.float32, [None, max_no_turns, num_slots], name="domain_labels")
    # dropout placeholder, 0.5 for training, 1.0 for validation/testing:
    keep_prob = tf.placeholder("float")

    # ------------------------------------ Create the Encoder networks ------------------------------------------------
    devs = ['/device:CPU:0']
    if dev == 'gpu':
        devs = get_available_devs()

    if net_type == "cnn":
        with tf.device(devs[1 % len(devs)]):
            # Encode the domain of the user input using a LSTM network
            usr_dom_en = define_CNN_model(user_input, num_filters=lstm_num_hidden, name="h_u_d")
            # Encode the domain of the system act using a LSTM network
            sys_dom_en = define_CNN_model(system_input, num_filters=lstm_num_hidden, name="h_s_d")

        with tf.device(devs[2 % len(devs)]):
            # Encode the slot of the user input using a CNN network
            usr_slot_en = define_CNN_model(user_input, num_filters=lstm_num_hidden, name="h_u_s")
            # Encode the slot of the system act using a CNN network
            sys_slot_en = define_CNN_model(system_input, num_filters=lstm_num_hidden, name="h_s_s")
            # Encode the value of the user input using a CNN network
            usr_val_en = define_CNN_model(user_input, num_filters=lstm_num_hidden, name="h_u_v")
            # Encode the value of the system act using a CNN network
            sys_val_en = define_CNN_model(system_input, num_filters=lstm_num_hidden, name="h_s_v")
            # Encode the user using a CNN network
            usr_en = define_CNN_model(user_input, num_filters=lstm_num_hidden // 5, name="h_u")

    else:

        with tf.device(devs[1 % len(devs)]):
            # Encode the domain of the user input using a LSTM network
            usr_dom_en = lstm_model(user_input, user_utterance_lengths, lstm_num_hidden, "h_u_d", net_type, bidir)
            usr_dom_en = tf.nn.dropout(usr_dom_en, keep_prob, name="h_u_d_out")
            # Encode the domain of the system act using a LSTM network
            sys_dom_en = lstm_model(system_input, sys_utterance_lengths, lstm_num_hidden, "h_s_d", net_type, bidir)
            sys_dom_en = tf.nn.dropout(sys_dom_en, keep_prob, name="h_s_d_out")

        with tf.device(devs[2 % len(devs)]):
            # Encode the slot of the user input using a LSTM network
            usr_slot_en = lstm_model(user_input, user_utterance_lengths, lstm_num_hidden, "h_u_s", net_type, bidir)
            usr_slot_en = tf.nn.dropout(usr_slot_en, keep_prob, name="h_u_s_out")
            # Encode the slot of the system act using a LSTM network
            sys_slot_en = lstm_model(system_input, sys_utterance_lengths, lstm_num_hidden, "h_s_s", net_type, bidir)
            sys_slot_en = tf.nn.dropout(sys_slot_en, keep_prob, name="h_s_s_out")
            # Encode the value of the user input using a LSTM network
            usr_val_en = lstm_model(user_input, user_utterance_lengths, lstm_num_hidden, "h_u_v", net_type, bidir)
            usr_val_en = tf.nn.dropout(usr_val_en, keep_prob, name="h_u_v_out")
            # Encode the value of the system act using a LSTM network
            sys_val_en = lstm_model(system_input, sys_utterance_lengths, lstm_num_hidden, "h_s_v", net_type, bidir)
            sys_val_en = tf.nn.dropout(sys_val_en, keep_prob, name="h_s_v_out")
            # Encode the user using a LSTM network
            usr_en = lstm_model(user_input, user_utterance_lengths, lstm_num_hidden // 5, "h_u", net_type, bidir)
            usr_en = tf.nn.dropout(usr_en, keep_prob, name="h_u_out")

    with tf.device(devs[1 % len(devs)]):
        usr_dom_en = tf.tile(tf.expand_dims(usr_dom_en, axis=2), [1, 1, num_slots, 1], name="h_u_d")
        sys_dom_en = tf.tile(tf.expand_dims(sys_dom_en, axis=2), [1, 1, num_slots, 1], name="h_s_d")
    with tf.device(devs[2 % len(devs)]):
        usr_slot_en = tf.tile(tf.expand_dims(usr_slot_en, axis=2), [1, 1, num_slots, 1], name="h_u_s")
        sys_slot_en = tf.tile(tf.expand_dims(sys_slot_en, axis=2), [1, 1, num_slots, 1], name="h_s_s")
        usr_val_en = tf.tile(tf.expand_dims(usr_val_en, axis=2), [1, 1, num_slots, 1], name="h_u_v")
        sys_val_en = tf.tile(tf.expand_dims(sys_val_en, axis=2), [1, 1, num_slots, 1], name="h_s_v")
        usr_en = tf.tile(tf.expand_dims(usr_en, axis=2), [1, 1, num_slots, 1], name="h_u")

    # All encoding vectors have size [batch_size, max_turns, num_slots, num_hidden]

    # Matrix that transforms the ontology from the embedding space to the hidden representation
    with tf.device(devs[1 % len(devs)]):
        W_onto_domain = tf.Variable(tf.random_normal([vector_dimension, lstm_num_hidden]), name="W_onto_domain")
        W_onto_slot = tf.Variable(tf.random_normal([vector_dimension, lstm_num_hidden]), name="W_onto_slot")
        W_onto_value = tf.Variable(tf.random_normal([vector_dimension, lstm_num_hidden]), name="W_onto_value")

        # And biases
        b_onto_domain = tf.Variable(tf.zeros([lstm_num_hidden]), name="b_onto_domain")
        b_onto_slot = tf.Variable(tf.zeros([lstm_num_hidden]), name="b_onto_slot")
        b_onto_value = tf.Variable(tf.zeros([lstm_num_hidden]), name="b_onto_value")

        # Apply the transformation from the embedding space of the ontology to the hidden space
        domain_vec = tf.slice(ontology, begin=[0, 0], size=[-1, vector_dimension])
        slot_vec = tf.slice(ontology, begin=[0, vector_dimension], size=[-1, vector_dimension])
        value_vec = tf.slice(ontology, begin=[0, 2 * vector_dimension], size=[-1, vector_dimension])
        # Each [num_slots, vector_dimension]
        d = tf.nn.dropout(tf.tanh(tf.matmul(domain_vec, W_onto_domain) + b_onto_domain), keep_prob, name="d")
        s = tf.nn.dropout(tf.tanh(tf.matmul(slot_vec, W_onto_slot) + b_onto_slot), keep_prob, name="s")
        v = tf.nn.dropout(tf.tanh(tf.matmul(value_vec, W_onto_value) + b_onto_value), keep_prob, name="v")
        # Each [num_slots, num_hidden]

        # Apply the comparison mechanism for all the user and system utterances and ontology values
        domain_user = tf.multiply(usr_dom_en, d, name="domain_user")
        domain_sys = tf.multiply(sys_dom_en, d, name="domain_sys")
        slot_user = tf.multiply(usr_slot_en, s, name="slot_user")
        slot_sys = tf.multiply(sys_slot_en, s, name="slot_sys")
        value_user = tf.multiply(usr_val_en, v, name="value_user")
        value_sys = tf.multiply(sys_val_en, v, name="value_sys")
        # All of size [batch_size, max_turns, num_slots, num_hidden]

        # -------------- Domain Detection -------------------------------------------------------------------------
        W_domain = tf.Variable(tf.random_normal([2 * lstm_num_hidden]), name="W_domain")
        b_domain = tf.Variable(tf.zeros([1]), name="b_domain")
        y_d = tf.sigmoid(tf.reduce_sum(tf.multiply(tf.concat([domain_user, domain_sys], axis=3), W_domain), axis=3)
                         + b_domain)  # [batch_size, max_turns, num_slots]

    # -------- Run through each of the 3 case ( inform, request, confirm) and decode the inferred state ---------
    # 1 Inform (User is informing the system about the goal, e.g. "I am looking for a place to stay in the centre")
    W_inform = tf.Variable(tf.random_normal([2 * lstm_num_hidden]), name="W_inform")
    b_inform = tf.Variable(tf.random_normal([1]), name="b_inform")
    inform = tf.add(tf.reduce_sum(tf.multiply(tf.concat([slot_user, value_user], axis=3), W_inform), axis=3), b_inform,
                    name="inform")  # [batch_size, max_turns, num_slots]

    # 2 Request (The system is requesting information from the user, e.g. "what type of food would you like?")
    with tf.device(devs[2 % len(devs)]):
        W_request = tf.Variable(tf.random_normal([2 * lstm_num_hidden]), name="W_request")
        b_request = tf.Variable(tf.random_normal([1]), name="b_request")
        request = tf.add(tf.reduce_sum(tf.multiply(tf.concat([slot_sys, value_user], axis=3), W_request), axis=3),
                         b_request, name="request")  # [batch_size, max_turns, num_slots]

    # 3 Confirm (The system is confirming values given by the user, e.g. "How about turkish food?")
    with tf.device(devs[3 % len(devs)]):
        size = 2 * lstm_num_hidden + lstm_num_hidden // 5
        W_confirm = tf.Variable(tf.random_normal([size]), name="W_confirm")
        b_confirm = tf.Variable(tf.random_normal([1]), name="b_confirm")
        confirm = tf.add(
            tf.reduce_sum(tf.multiply(tf.concat([slot_sys, value_sys, usr_en], axis=3), W_confirm), axis=3),
            b_confirm, name="confirm")  # [batch_size, max_turns, num_slots]

    output = inform + request + confirm

    # -------------------- Adding the belief update RNN with memory cell (Taken from previous model) -------------------
    with tf.device(devs[2 % len(devs)]):
        domain_memory = tf.Variable(tf.random_normal([1, 1]), name="domain_memory")
        domain_current = tf.Variable(tf.random_normal([1, 1]), name="domain_current")
        domain_M_h = tf.Variable(tf.random_normal([1, 1]), name="domain_M_h")
        domain_W_m = tf.Variable(tf.random_normal([1, 1], name="domain_W_m"))
        domain_U_m = tf.Variable(tf.random_normal([1, 1]), name="domain_U_m")
    a_memory = tf.Variable(tf.random_normal([1, 1]), name="a_memory")
    b_memory = tf.Variable(tf.random_normal([1, 1]), name="b_memory")
    a_current = tf.Variable(tf.random_normal([1, 1]), name="a_current")
    b_current = tf.Variable(tf.random_normal([1, 1]), name="b_current")
    M_h_a = tf.Variable(tf.random_normal([1, 1]), name="M_h_a")
    M_h_b = tf.Variable(tf.random_normal([1, 1]), name="M_h_b")
    W_m_a = tf.Variable(tf.random_normal([1, 1]), name="W_m_a")
    W_m_b = tf.Variable(tf.random_normal([1, 1]), name="W_m_b")
    U_m_a = tf.Variable(tf.random_normal([1, 1]), name="U_m_a")
    U_m_b = tf.Variable(tf.random_normal([1, 1]), name="U_m_b")

    # ---------------------------------- Unroll the domain over time --------------------------------------------------
    with tf.device(devs[1 % len(devs)]):
        cell = GRU(domain_memory * tf.diag(tf.ones(num_slots)), domain_current * tf.diag(tf.ones(num_slots)),
                   domain_M_h * tf.diag(tf.ones(num_slots)), domain_W_m * tf.diag(tf.ones(num_slots)),
                   domain_U_m * tf.diag(tf.ones(num_slots)), num_slots,
                   binary_output=True)

        y_d, _ = tf.nn.dynamic_rnn(cell, y_d, sequence_length=num_turns, dtype=tf.float32)

        domain_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=domain_labels, logits=y_d), axis=2,
                                    name="domain_loss") / (num_slots / len(slots))

        y_d = tf.sigmoid(y_d)

    with tf.device(devs[0 % len(devs)]):

        loss = [None for _ in range(len(slots))]
        slot_pred = [None for _ in range(len(slots))]
        slot_label = [None for _ in range(len(slots))]
        val_pred = [None for _ in range(len(slots))]
        val_label = [None for _ in range(len(slots))]
        y = [None for _ in range(len(slots))]
        y_pred = [None for _ in range(len(slots))]
        for i in range(len(slots)):

            num_values = slots[i] + 1  # For the none case
            size = sum(slots[:i + 1]) - slots[i]
            if test:
                domain_output = tf.slice(tf.round(y_d), begin=[0, 0, size], size=[-1, -1, slots[i]])
            else:
                domain_output = tf.slice(domain_labels, begin=[0, 0, size], size=[-1, -1, slots[i]])
            max_val = tf.expand_dims(tf.reduce_max(domain_output, axis=2), axis=2)
            # tf.assert_less_equal(max_val, 1.0)
            # tf.assert_equal(tf.round(max_val), max_val)
            domain_output = tf.concat([tf.zeros(tf.shape(domain_output)), 1 - max_val], axis=2)

            slot_output = tf.slice(output, begin=[0, 0, size], size=[-1, -1, slots[i]])
            slot_output = tf.concat([slot_output, tf.zeros([tf.shape(output)[0], max_no_turns, 1])], axis=2)

            labels_output = tf.slice(labels, begin=[0, 0, size], size=[-1, -1, slots[i]])
            max_val = tf.expand_dims(tf.reduce_max(labels_output, axis=2), axis=2)
            # tf.assert_less_equal(max_val, 1.0)
            # tf.assert_equal(tf.round(max_val), max_val)
            slot_label[i] = max_val
            # [Batch_size, max_turns, 1]
            labels_output = tf.argmax(tf.concat([labels_output, 1 - max_val], axis=2), axis=2)
            # [Batch_size, max_turns]
            val_label[i] = tf.cast(tf.expand_dims(labels_output, axis=2), dtype="float")
            # [Batch_size, max_turns, 1]

            diag_memory = a_memory * tf.diag(tf.ones(num_values))
            non_diag_memory = tf.matrix_set_diag(b_memory * tf.ones([num_values, num_values]), tf.zeros(num_values))
            W_memory = diag_memory + non_diag_memory

            diag_current = a_current * tf.diag(tf.ones(num_values))
            non_diag_current = tf.matrix_set_diag(b_current * tf.ones([num_values, num_values]), tf.zeros(num_values))
            W_current = diag_current + non_diag_current

            diag_M_h = M_h_a * tf.diag(tf.ones(num_values))
            non_diag_M_h = tf.matrix_set_diag(M_h_b * tf.ones([num_values, num_values]), tf.zeros(num_values))
            M_h = diag_M_h + non_diag_M_h

            diag_U_m = U_m_a * tf.diag(tf.ones(num_values))
            non_diag_U_m = tf.matrix_set_diag(U_m_b * tf.ones([num_values, num_values]), tf.zeros(num_values))
            U_m = diag_U_m + non_diag_U_m

            diag_W_m = W_m_a * tf.diag(tf.ones(num_values))
            non_diag_W_m = tf.matrix_set_diag(W_m_b * tf.ones([num_values, num_values]), tf.zeros(num_values))
            W_m = diag_W_m + non_diag_W_m

            cell = GRU(W_memory, W_current, M_h, W_m, U_m, num_values)
            y_predict, _ = tf.nn.dynamic_rnn(cell, slot_output, sequence_length=num_turns, dtype=tf.float32)

            y_predict = y_predict + 1000000.0 * domain_output
            # [Batch_size, max_turns, num_values]

            y[i] = tf.nn.softmax(y_predict)
            val_pred[i] = tf.cast(tf.expand_dims(tf.argmax(y[i], axis=2), axis=2), dtype="float32")
            # [Batch_size, max_turns, 1]
            y_pred[i] = tf.slice(tf.one_hot(tf.argmax(y[i], axis=2), dtype=tf.float32, depth=num_values),
                                 begin=[0, 0, 0], size=[-1, -1, num_values - 1])
            y[i] = tf.slice(y[i], begin=[0, 0, 0], size=[-1, -1, num_values - 1])
            slot_pred[i] = tf.cast(tf.reduce_max(y_pred[i], axis=2, keep_dims=True), dtype="float32")
            # [Batch_size, max_turns, 1]
            loss[i] = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_output, logits=y_predict)
            # [Batch_size, max_turns]

    # ---------------- Compute the output and the loss function (cross_entropy) and add to optimizer--------------------
    cross_entropy = tf.add_n(loss, name="cross_entropy")
    # Add the error from the domains
    cross_entropy = tf.add(cross_entropy, domain_loss, name="total_loss")

    y = tf.concat(y, axis=2, name="y")

    mask = tf.cast(tf.sequence_mask(num_turns, maxlen=max_no_turns), dtype=tf.float32)
    mask_extended = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, num_slots])
    cross_entropy = tf.reduce_sum(mask * cross_entropy, axis=1) / tf.cast(num_turns, dtype=tf.float32)

    optimizer = tf.train.AdamOptimizer(0.001)
    train_step = optimizer.minimize(cross_entropy, colocate_gradients_with_ops=True)

    # ----------------- Get the precision, recall f1-score and accuracy -----------------------------------------------

    # Domain accuracy
    true_predictions = tf.reshape(domain_labels, [-1, num_slots])
    predictions = tf.reshape(tf.round(y_d) * mask_extended, [-1, num_slots])

    y_d = tf.reshape(y_d * mask_extended, [-1, num_slots])

    _, _, _, domain_accuracy = get_metrics(predictions, true_predictions, num_turns, mask_extended, num_slots)

    mask_extended_2 = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, len(slots)])

    # Slot accuracy
    true_predictions = tf.reshape(tf.concat(slot_label, axis=2), [-1, len(slots)])
    predictions = tf.reshape(tf.concat(slot_pred, axis=2) * mask_extended_2, [-1, len(slots)])

    _, _, _, slot_accuracy = get_metrics(predictions, true_predictions, num_turns, mask_extended_2, len(slots))

    # accuracy
    if test:
        value_accuracy = []
        mask_extended_3 = tf.expand_dims(mask, axis=2)
        for i in range(len(slots)):
            true_predictions = tf.reshape(val_label[i] * mask_extended_3, [-1, 1])
            predictions = tf.reshape(val_pred[i] * mask_extended_3, [-1, 1])

            _, _, _, value_acc = get_metrics(predictions, true_predictions, num_turns, mask_extended_3, 1)
            value_accuracy.append(value_acc)

        value_accuracy = tf.stack(value_accuracy)
    else:
        true_predictions = tf.reshape(tf.concat(val_label, axis=2) * mask_extended_2, [-1, len(slots)])
        predictions = tf.reshape(tf.concat(val_pred, axis=2) * mask_extended_2, [-1, len(slots)])

        _, _, _, value_accuracy = get_metrics(predictions, true_predictions, num_turns, mask_extended_2, len(slots))

    # Value f1score a
    true_predictions = tf.reshape(labels, [-1, num_slots])
    predictions = tf.reshape(tf.concat(y_pred, axis=2) * mask_extended, [-1, num_slots])

    precision, recall, value_f1_score, _ = get_metrics(predictions, true_predictions, num_turns,
                                                       mask_extended, num_slots)

    y_ = tf.reshape(y, [-1, num_slots])

    # -------------------- Summarise the statistics of training to be viewed in tensorboard-----------------------------
    tf.summary.scalar("domain_accuracy", domain_accuracy)
    tf.summary.scalar("slot_accuracy", slot_accuracy)
    tf.summary.scalar("value_accuracy", value_accuracy)
    tf.summary.scalar("value_f1_score", value_f1_score)
    tf.summary.scalar("cross_entropy", tf.reduce_mean(cross_entropy))

    value_f1_score = [precision, recall, value_f1_score]

    return user_input, system_input, num_turns, user_utterance_lengths, sys_utterance_lengths, labels, domain_labels, \
           domain_accuracy, slot_accuracy, value_accuracy, value_f1_score, train_step, keep_prob, predictions, \
           true_predictions, [y_, y_d]


def get_metrics(predictions, true_predictions, no_turns, mask, num_slots):
    mask = tf.reshape(mask, [-1, num_slots])
    correct_prediction = tf.cast(tf.equal(predictions, true_predictions), "float32") * mask

    num_positives = tf.reduce_sum(true_predictions)
    classified_positives = tf.reduce_sum(predictions)

    true_positives = tf.multiply(predictions, true_predictions)
    num_true_positives = tf.reduce_sum(true_positives)

    recall = num_true_positives / num_positives
    precision = num_true_positives / classified_positives
    f_score = (2 * recall * precision) / (recall + precision)
    accuracy = tf.reduce_sum(correct_prediction) / (tf.cast(tf.reduce_sum(no_turns), dtype="float32") * num_slots)

    return precision, recall, f_score, accuracy



# main.py
def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt(sum(word_vectors[word]**2) + 1e-6)
        word_vectors[word] *= norm
    return word_vectors


def xavier_vector(word, D=300):
    """
    Returns a D-dimensional vector for the word.

    We hash the word to always get the same vector for the given word.
    """
    def hash_string(_s):
        return abs(hash(_s)) % (10 ** 8)
    seed_value = hash_string(word)
    np.random.seed(seed_value)

    neg_value = - math.sqrt(6)/math.sqrt(D)
    pos_value = math.sqrt(6)/math.sqrt(D)

    rsample = np.random.uniform(low=neg_value, high=pos_value, size=(D,))
    norm = np.linalg.norm(rsample)
    rsample_normed = rsample/norm

    return rsample_normed


def load_ontology(url, word_vectors):
    '''Load the ontology from a file

    :param url: to the ontology
    :param word_vectors: dictionary of the word embeddings [words, vector_dimension]
    :return: list([domain-slot-value]), [no_slots, vector_dimension]
    '''
    global num_slots
    # print("\tMDBT: Loading the ontology....................")
    data = json.load(open(url, mode='r', encoding='utf8'), object_pairs_hook=OrderedDict)
    slot_values = []
    ontology = []
    slots_values = []
    ontology_vectors = []
    for slots in data:
        [domain, slot] = slots.split('-')
        if domain not in domains or slot == 'name':
            continue
        values = data[slots]
        if "book" in slot:
            [slot, value] = slot.split(" ")
            booking_slots[domain+'-'+value] = values
            values = [value]
        elif slot == "departure" or slot == "destination":
            values = ["place"]
        domain_vec = np.sum(process_text(domain, word_vectors), axis=0)
        if domain not in word_vectors:
            word_vectors[domain.replace(" ", "")] = domain_vec
        slot_vec = np.sum(process_text(slot, word_vectors), axis=0)
        if domain+'-'+slot not in slots_values:
            slots_values.append(domain+'-'+slot)
        if slot not in word_vectors:
            word_vectors[slot.replace(" ", "")] = slot_vec
        slot_values.append(len(values))
        for value in values:
            ontology.append(domain + '-' + slot + '-' + value)
            value_vec = np.sum(process_text(value, word_vectors, print_mode=True), axis=0)
            if value not in word_vectors:
                word_vectors[value.replace(" ", "")] = value_vec
            ontology_vectors.append(np.concatenate((domain_vec, slot_vec, value_vec)))

    num_slots = len(slots_values)
    # print("\tMDBT: We have about {} values".format(len(ontology)))
    # print("\tMDBT: The Full Ontology is:")
    # print(ontology)
    # print("\tMDBT: The slots in this ontology:")
    # print(slots_values)
    return ontology, np.asarray(ontology_vectors, dtype='float32'), slot_values


def load_word_vectors(url):
    '''Load the word embeddings from the url

    :param url: to the word vectors
    :return: dict of word and vector values
    '''
    word_vectors = {}
    # print("Loading the word embeddings....................")
    # print('abs path: ', os.path.abspath(url))
    with open(url, mode='r', encoding='utf8') as f:
        for line in f:
            line = line.split(" ", 1)
            key = line[0]
            word_vectors[key] = np.fromstring(line[1], dtype="float32", sep=" ")
    # print("\tMDBT: The vocabulary contains about {} word embeddings".format(len(word_vectors)))
    return normalise_word_vectors(word_vectors)


def track_dialogue(data, ontology, predictions, y):
    overall_accuracy_total = 0
    overall_accuracy_corr = 0
    joint_accuracy_total = 0
    joint_accuracy_corr = 0
    global num_slots
    dialogues = []
    idx = 0
    for dialogue in data:
        turn_ids = []
        for key in dialogue.keys():
            if key.isdigit():
                turn_ids.append(int(key))
            elif dialogue[key] and key not in domains:
                continue
        turn_ids.sort()
        turns = []
        previous_terms = []
        for key in turn_ids:
            turn = dialogue[str(key)]
            user_input = turn['user']['text']
            sys_res = turn['system']
            state = turn['user']['belief_state']
            turn_obj = dict()
            turn_obj['user'] = user_input
            turn_obj['system'] = sys_res
            prediction = predictions[idx, :]
            indices = np.argsort(prediction)[:-(int(np.sum(prediction)) + 1):-1]
            predicted_terms = [process_booking(ontology[i], user_input, previous_terms) for i in indices]
            previous_terms = deepcopy(predicted_terms)
            turn_obj['prediction'] = ["{}: {}".format(predicted_terms[x], y[idx, i]) for x, i in enumerate(indices)]
            turn_obj['True state'] = []
            idx += 1
            unpredicted_labels = 0
            for domain in state:
                if domain not in domains:
                    continue
                slots = state[domain]['semi']
                for slot in slots:
                    if slot == 'name':
                        continue
                    value = slots[slot]
                    if value != '':
                        label = domain + '-' + slot + '-' + value
                        turn_obj['True state'].append(label)
                        if label in predicted_terms:
                            predicted_terms.remove(label)
                        else:
                            unpredicted_labels += 1

            turns.append(turn_obj)
            overall_accuracy_total += num_slots
            overall_accuracy_corr += (num_slots - unpredicted_labels - len(predicted_terms))
            if unpredicted_labels + len(predicted_terms) == 0:
                joint_accuracy_corr += 1
            joint_accuracy_total += 1

        dialogues.append(turns)
    return dialogues, overall_accuracy_corr/overall_accuracy_total, joint_accuracy_corr/joint_accuracy_total


def process_booking(ontolog_term, usr_input, previous_terms):
    usr_input = usr_input.lower().split()
    domain, slot, value = ontolog_term.split('-')
    if slot == 'book':
        for term in previous_terms:
            if domain+'-book '+value in term:
                ontolog_term = term
                break
        else:
            if value == 'stay' or value == 'people':
                numbers = [int(s) for s in usr_input if s.isdigit()]
                if len(numbers) == 1:
                    ontolog_term = domain + '-' + slot + ' ' + value + '-' + str(numbers[0])
                elif len(numbers) == 2:
                    vals = {}
                    if usr_input[usr_input.index(str(numbers[0]))+1] in ['people', 'person']:
                        vals['people'] = str(numbers[0])
                        vals['stay'] = str(numbers[1])
                    else:
                        vals['people'] = str(numbers[1])
                        vals['stay'] = str(numbers[0])
                    ontolog_term = domain + '-' + slot + ' ' + value + '-' + vals[value]
            else:
                for val in booking_slots[domain+'-'+value]:
                    if val in ' '.join(usr_input):
                        ontolog_term = domain + '-' + slot + ' ' + value + '-' + val
                        break
    return ontolog_term


def process_history(sessions, word_vectors, ontology):
    '''Load the woz3 data and extract feature vectors

    :param data: the data to load
    :param word_vectors: word embeddings
    :param ontology: list of domain-slot-value
    :param url: Is the data coming from a url, default true
    :return: list(num of turns, user_input vectors, system_response vectors, labels)
    '''
    dialogues = []
    actual_dialogues = []
    for dialogue in sessions:
        turn_ids = []
        for key in dialogue.keys():
            if key.isdigit():
                turn_ids.append(int(key))
            elif dialogue[key] and key not in domains:
                continue
        turn_ids.sort()
        num_turns = len(turn_ids)
        user_vecs = []
        sys_vecs = []
        turn_labels = []
        turn_domain_labels = []
        add = False
        good = True
        pre_sys = np.zeros([max_utterance_length, vector_dimension], dtype="float32")
        for key in turn_ids:
            turn = dialogue[str(key)]
            user_v, sys_v, labels, domain_labels = process_turn(turn, word_vectors, ontology)
            if good and (user_v.shape[0] > max_utterance_length or pre_sys.shape[0] > max_utterance_length):
                # cut overlength utterance instead of discarding them
                if user_v.shape[0] > max_utterance_length:
                    user_v = user_v[:max_utterance_length]
                if pre_sys.shape[0] > max_utterance_length:
                    pre_sys = pre_sys[:max_utterance_length]
                # good = False
                # break
            user_vecs.append(user_v)
            sys_vecs.append(pre_sys)
            turn_labels.append(labels)
            turn_domain_labels.append(domain_labels)
            if not add and sum(labels) > -1:
                add = True
            pre_sys = sys_v
        if add and good:
            dialogues.append((num_turns, user_vecs, sys_vecs, turn_labels, turn_domain_labels))
            actual_dialogues.append(dialogue)
    # print("\tMDBT: The data contains about {} dialogues".format(len(dialogues)))
    return dialogues, actual_dialogues

def load_woz_data_new(data, word_vectors, ontology, url=False):
    '''Ported from load_woz_data, using tatk.util.dataloader pkg

    :param data: the data to load
    :param word_vectors: word embeddings
    :param ontology: list of domain-slot-value
    :param url: Is the data coming from a url, default true
    :return: list(num of turns, user_input vectors, system_response vectors, labels)
    '''
    if url:
        data = json.load(open(url, mode='r', encoding='utf8'))
    dialogues = []
    actual_dialogues = []
    for dialogue in data:
        turn_ids = []
        for key in dialogue.keys():
            if key.isdigit():
                turn_ids.append(int(key))
            elif dialogue[key] and key not in domains:
                continue
        turn_ids.sort()
        num_turns = len(turn_ids)
        user_vecs = []
        sys_vecs = []
        turn_labels = []
        turn_domain_labels = []
        add = False
        good = True
        pre_sys = np.zeros([max_utterance_length, vector_dimension], dtype="float32")
        for key in turn_ids:
            turn = dialogue[str(key)]
            user_v, sys_v, labels, domain_labels = process_turn(turn, word_vectors, ontology)
            if good and (user_v.shape[0] > max_utterance_length or pre_sys.shape[0] > max_utterance_length):
                good = False
                break
            user_vecs.append(user_v)
            sys_vecs.append(pre_sys)
            turn_labels.append(labels)
            turn_domain_labels.append(domain_labels)
            if not add and sum(labels) > 0:
                add = True
            pre_sys = sys_v
        if add and good:
            dialogues.append((num_turns, user_vecs, sys_vecs, turn_labels, turn_domain_labels))
            actual_dialogues.append(dialogue)
    # print("\tMDBT: The data contains about {} dialogues".format(len(dialogues)))
    return dialogues, actual_dialogues

def load_woz_data(data, word_vectors, ontology, url=True):
    '''Load the woz3 data and extract feature vectors

    :param data: the data to load
    :param word_vectors: word embeddings
    :param ontology: list of domain-slot-value
    :param url: Is the data coming from a url, default true
    :return: list(num of turns, user_input vectors, system_response vectors, labels)
    '''
    if url:
        # print("Loading data from url {} ....................".format(data))
        data = json.load(open(data, mode='r', encoding='utf8'))

    dialogues = []
    actual_dialogues = []
    for dialogue in data:
        turn_ids = []
        for key in dialogue.keys():
            if key.isdigit():
                turn_ids.append(int(key))
            elif dialogue[key] and key not in domains:
                continue
        turn_ids.sort()
        num_turns = len(turn_ids)
        user_vecs = []
        sys_vecs = []
        turn_labels = []
        turn_domain_labels = []
        add = False
        good = True
        pre_sys = np.zeros([max_utterance_length, vector_dimension], dtype="float32")
        for key in turn_ids:
            turn = dialogue[str(key)]
            user_v, sys_v, labels, domain_labels = process_turn(turn, word_vectors, ontology)
            if good and (user_v.shape[0] > max_utterance_length or pre_sys.shape[0] > max_utterance_length):
                good = False
                break
            user_vecs.append(user_v)
            sys_vecs.append(pre_sys)
            turn_labels.append(labels)
            turn_domain_labels.append(domain_labels)
            if not add and sum(labels) > 0:
                add = True
            pre_sys = sys_v
        if add and good:
            dialogues.append((num_turns, user_vecs, sys_vecs, turn_labels, turn_domain_labels))
            actual_dialogues.append(dialogue)
    # print("\tMDBT: The data contains about {} dialogues".format(len(dialogues)))
    return dialogues, actual_dialogues


def process_turn(turn, word_vectors, ontology):
    '''Process a single turn extracting and processing user text, system response and labels

    :param turn: dict
    :param word_vectors: word embeddings
    :param ontology: list(domain-slot-value)
    :return: ([utterance length, 300], [utterance length, 300], [no_slots])
    '''
    user_input = turn['user']['text']
    sys_res = turn['system']
    state = turn['user']['belief_state']
    user_v = process_text(user_input, word_vectors, ontology)
    sys_v = process_text(sys_res, word_vectors, ontology)
    labels = np.zeros(len(ontology), dtype='float32')
    domain_labels = np.zeros(len(ontology), dtype='float32')
    for domain in state:
        if domain not in domains:
            continue
        slots = state[domain]['semi']
        domain_mention = False
        for slot in slots:

            if slot == 'name':
                continue
            value = slots[slot]
            if "book" in slot:
                [slot, value] = slot.split(" ")
            if value != '' and value != 'corsican':
                if slot == "destination" or slot == "departure":
                    value = "place"
                elif value == '09;45':
                    value = '09:45'
                elif 'alpha-milton' in value:
                    value = value.replace('alpha-milton', 'alpha milton')
                elif value == 'east side':
                    value = 'east'
                elif value == ' expensive':
                    value = 'expensive'
                labels[ontology.index(domain + '-' + slot + '-' + value)] = 1
                domain_mention = True
        if domain_mention:
            for idx, slot in enumerate(ontology):
                if domain in slot:
                    domain_labels[idx] = 1

    return user_v, sys_v, labels, domain_labels


def process_text(text, word_vectors, ontology=None, print_mode=False):
    '''Process a line/sentence converting words to feature vectors

    :param text: sentence
    :param word_vectors: word embeddings
    :param ontology: The ontology to do exact matching
    :param print_mode: Log the cases where the word is not in the pre-trained word vectors
    :return: [length of sentence, 300]
    '''
    text = text.replace("(", "").replace(")", "").replace('"', "").replace(u"’", "'").replace(u"‘", "'")
    text = text.replace("\t", "").replace("\n", "").replace("\r", "").strip().lower()
    text = text.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('-', ' ').replace('/', ' / ')\
        .replace(':', ' ')
    if ontology:
        for slot in ontology:
            [domain, slot, value] = slot.split('-')
            text.replace(domain, domain.replace(" ", ""))\
                .replace(slot, slot.replace(" ", ""))\
                .replace(value, value.replace(" ", ""))

    words = text.split()

    vectors = []
    for word in words:
        word = word.replace("'", "").replace("!", "")
        if word == "":
            continue
        if word not in word_vectors:
            length = len(word)
            for i in range(1, length)[::-1]:
                if word[:i] in word_vectors and word[i:] in word_vectors:
                    vec = word_vectors[word[:i]] + word_vectors[word[i:]]
                    break
            else:
                vec = xavier_vector(word)
                word_vectors[word] = vec
                if print_mode:
                    pass
                    # print("\tMDBT: Adding new word: {}".format(word))
        else:
            vec = word_vectors[word]
        vectors.append(vec)
    return np.asarray(vectors, dtype='float32')


def generate_batch(dialogues, batch_no, batch_size, ontology_size):
    '''Generate examples for minibatch training

    :param dialogues: list(num of turns, user_input vectors, system_response vectors, labels)
    :param batch_no: where we are in the training data
    :param batch_size: number of dialogues to generate
    :param ontology_size: no_slots
    :return: list(user_input, system_response, labels, user_sentence_length, system_sentence_length, number of turns)
    '''
    user = np.zeros((batch_size, max_no_turns, max_utterance_length, vector_dimension), dtype='float32')
    sys_res = np.zeros((batch_size, max_no_turns, max_utterance_length, vector_dimension), dtype='float32')
    labels = np.zeros((batch_size, max_no_turns, ontology_size), dtype='float32')
    domain_labels = np.zeros((batch_size, max_no_turns, ontology_size), dtype='float32')
    user_uttr_len = np.zeros((batch_size, max_no_turns), dtype='int32')
    sys_uttr_len = np.zeros((batch_size, max_no_turns), dtype='int32')
    no_turns = np.zeros(batch_size, dtype='int32')
    idx = 0
    for i in range(batch_no*train_batch_size, batch_no*train_batch_size + batch_size):
        (num_turns, user_vecs, sys_vecs, turn_labels, turn_domain_labels) = dialogues[i]
        no_turns[idx] = num_turns
        for j in range(num_turns):
            user_uttr_len[idx, j] = user_vecs[j].shape[0]
            sys_uttr_len[idx, j] = sys_vecs[j].shape[0]
            user[idx, j, :user_uttr_len[idx, j], :] = user_vecs[j]
            sys_res[idx, j, :sys_uttr_len[idx, j], :] = sys_vecs[j]
            labels[idx, j, :] = turn_labels[j]
            domain_labels[idx, j, :] = turn_domain_labels[j]
        idx += 1
    return user, sys_res, labels, domain_labels, user_uttr_len, sys_uttr_len, no_turns


def evaluate_model(sess, model_variables, val_data, summary, batch_id, i):

    '''Evaluate the model against validation set

    :param sess: training session
    :param model_variables: all model input variables
    :param val_data: validation data
    :param summary: For tensorboard
    :param batch_id: where we are in the training data
    :param i: the index of the validation data to load
    :return: evaluation accuracy and the summary
    '''

    (user, sys_res, no_turns, user_uttr_len, sys_uttr_len, labels, domain_labels, domain_accuracy,
     slot_accuracy, value_accuracy, value_f1, train_step, keep_prob, _, _, _) = model_variables

    batch_user, batch_sys, batch_labels, batch_domain_labels, batch_user_uttr_len, batch_sys_uttr_len, \
        batch_no_turns = val_data

    start_time = time.time()

    b_z = train_batch_size
    [precision, recall, value_f1] = value_f1
    [d_acc, s_acc, v_acc, f1_score, pr, re, sm1, sm2] = sess.run([domain_accuracy, slot_accuracy, value_accuracy,
                                                                  value_f1, precision, recall] + summary,
                                                           feed_dict={user: batch_user[i:i+b_z, :, :, :],
                                                                      sys_res: batch_sys[i:i+b_z, :, :, :],
                                                                      labels: batch_labels[i:i+b_z, :, :],
                                                                      domain_labels: batch_domain_labels[i:i+b_z, :, :],
                                                                      user_uttr_len: batch_user_uttr_len[i:i+b_z, :],
                                                                      sys_uttr_len: batch_sys_uttr_len[i:i+b_z, :],
                                                                      no_turns: batch_no_turns[i:i+b_z],
                                                                      keep_prob: 1.0})

    print("Batch", batch_id, "[Domain Accuracy] = ", d_acc, "[Slot Accuracy] = ", s_acc, "[Value Accuracy] = ",
          v_acc, "[F1 Score] = ", f1_score, "[Precision] = ", pr, "[Recall] = ", re,
          " ----- ", round(time.time() - start_time, 3),
          "seconds. ---")

    return d_acc, s_acc, v_acc, f1_score, sm1, sm2
