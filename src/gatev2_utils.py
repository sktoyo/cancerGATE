import numpy as np
import tensorflow as tf


import random
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import dgl


def get_reverse_edge(edge_list):
    return np.array([[edge[1], edge[0]] for edge in edge_list])


def check_input_argue(input_argue, exp_input, mut_input):
    if input_argue == 'mutation':
        in_dims = [mut_input.shape[1]]
        train_input = [mut_input]
    elif input_argue == 'expression':
        in_dims = [exp_input.shape[1]]
        train_input = [exp_input]
    elif input_argue == 'whole':
        in_dims = [exp_input.shape[1], mut_input.shape[1]]
        train_input = [exp_input, mut_input]
    else:
        in_dims = None
        train_input = None
    return in_dims, train_input


def convert_sparse_matrix_to_sparse_tensor(inputs):
    coo = inputs.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    output = tf.SparseTensor(indices, coo.data.astype('float64'), coo.shape)
    output = tf.dtypes.cast(output, tf.float32)
    return output


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


# create network - from train edges
def create_input_network(edge_list, gpu_usage=True):
    humannet_dgl = dgl.graph((edge_list[:, 0], edge_list[:, 1]))
    if gpu_usage:
        humannet_dgl = humannet_dgl.to('/gpu:0')
    else:
        pass
    g = humannet_dgl

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g


def data_split(adj, validation_rate,
               test_rate):
    coords, values, shape = sparse_to_tuple(adj)

    # create positive set for train, validation, test from
    coords = coords.tolist()
    positive_set = np.array([coo for coo in coords if coo[0] < coo[1]])
    positive_idx = np.array([coo[0] * shape[0] + coo[1] for coo in positive_set])

    np.random.shuffle(positive_set)

    test_num = int(len(positive_set) * test_rate)
    validation_num = int(len(positive_set) * validation_rate)
    test_pos = positive_set[:test_num]
    valid_pos = positive_set[test_num:(test_num + validation_num)]
    train_edges = positive_set[(test_num + validation_num):]

    # create negative set for validation, test
    negative_idx_list = list()

    while len(negative_idx_list) < len(positive_idx):
        i = random.randrange(shape[0])
        j = random.randrange(shape[0])
        if i < j:
            idx = i * shape[0] + j
            if idx not in positive_idx:
                negative_idx_list.append(idx)

    negative_idx = np.array(negative_idx_list)
    negative_set = np.array([[idx // shape[0], idx % shape[0]] for idx in negative_idx])
    test_neg = negative_set[:test_num]
    valid_neg = negative_set[test_num:(test_num + validation_num)]

    return positive_set, negative_set, train_edges, valid_pos, valid_neg, test_pos, test_neg

def accuracy_ae(logits, labels):
    # acc function
    correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(logits), 0.5), tf.float32),
                                  tf.cast(labels, tf.float32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc.numpy().item()


def accuracy_cls(logits, labels):
    indices = tf.math.argmax(logits, axis=1)
    label_indices = tf.math.argmax(labels, axis=1)
    acc = tf.reduce_mean(tf.cast(indices == label_indices, dtype=tf.float32))
    return acc.numpy().item()


def evaluate_ae(model, features, labels):
    logits = model(features, training=False)
    return accuracy_ae(logits, labels)


def evaluate_cls(model, features, labels, mask):
    logits = model(features, training=False)
    logits = logits[mask]
    labels = labels[mask]
    return accuracy_cls(logits, labels)


def get_roc_score(model, train_input, edges_pos, edges_neg):
    logits_pos = tf.sigmoid(model.get_reconstructed_edge(train_input, edges_pos)).numpy()
    logits_neg = tf.sigmoid(model.get_reconstructed_edge(train_input, edges_neg)).numpy()

    preds_all = np.hstack([logits_pos, logits_neg])
    labels_all = np.hstack([np.ones(len(logits_pos)), np.zeros(len(logits_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_roc_score2(model, g, feature, edges_pos, edges_neg):
    logits_pos = tf.sigmoid(model.get_reconstructed_edge(g, feature, edges_pos)).numpy()
    logits_neg = tf.sigmoid(model.get_reconstructed_edge(g, feature, edges_neg)).numpy()

    preds_all = np.hstack([logits_pos, logits_neg])
    labels_all = np.hstack([np.ones(len(logits_pos)), np.zeros(len(logits_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score