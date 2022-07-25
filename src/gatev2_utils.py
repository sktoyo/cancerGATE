import numpy as np
import tensorflow as tf

import pandas as pd
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


def get_pos_only_dataset(oncokb_unique_index, ncg_index, total_negative_index):
    random.shuffle(total_negative_index)
    train_ratio = 1 - len(oncokb_unique_index) / len(ncg_index)
    train_count = int(train_ratio * len(total_negative_index))
    pos_only_train_negative_index = total_negative_index[:train_count]
    pos_only_test_negative_index = total_negative_index[train_count:]

    pos_only_train_index_list = ncg_index + pos_only_train_negative_index
    pos_only_train_label_list = np.concatenate((np.ones(len(ncg_index)), np.zeros(len(pos_only_train_negative_index))))
    pos_only_test_index_list = oncokb_unique_index + pos_only_test_negative_index
    pos_only_test_label_list = np.concatenate(
        (np.ones(len(oncokb_unique_index)), np.zeros(len(pos_only_test_negative_index))))

    return pos_only_train_index_list, pos_only_train_label_list, pos_only_test_index_list, pos_only_test_label_list


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


def get_symbol_list():
    # target list - oncoKB + cosmic + literature
    with open("../data/gene_list/target and driver genes_total.csv", 'r') as f:
        genes = f.readlines()
        oncokb_list = [gene.rstrip() for gene in genes]

    # target list - NCGv7
    ncg_df = pd.read_csv('../data/gene_list/NCG7_cancerdrivers_breast.tsv', sep="\t")
    ncg_df = ncg_df.loc[~ncg_df['NCG_oncogene'].isna()]  # filter nan annotation
    ncg_symbol = ncg_df['symbol'].to_list()
    ncg_symbol = list(set(ncg_symbol))
    ncg_symbol.sort()

    # target list - negative set
    negative_df = pd.read_csv('../data/gene_list/cancer negative gene set.tsv', sep="\t",
                              index_col=False)
    negative_symbol = list(set(negative_df['symbol']))
    random.shuffle(negative_symbol)

    oncokb_unique = set(oncokb_list) - set(ncg_symbol)
    oncokb_unique = list(oncokb_unique)
    return ncg_symbol, oncokb_unique, negative_symbol


def get_index_list(ncg_symbol, oncokb_unique, negative_symbol):
    gene_index = pd.read_csv('preprocessing/gene_index.tsv', sep='\t', index_col=0)
    gene_list = gene_index.index.to_list()

    train_index_list = list()
    train_label_list = list()
    for i in range(len(gene_list)):
        symbol = gene_list[i]
        if symbol in ncg_symbol:
            train_index_list.append(i)
            train_label_list.append(1)
        elif symbol in negative_symbol:
            train_index_list.append(i)
            train_label_list.append(0)

    test_index_list = list()
    test_label_list = list()
    for i in range(len(gene_list)):
        symbol = gene_list[i]
        if symbol in oncokb_unique:
            test_index_list.append(i)
            test_label_list.append(1)
        elif symbol not in ncg_symbol and symbol not in negative_symbol:
            test_index_list.append(i)
            test_label_list.append(0)

    return train_index_list, train_label_list, test_index_list, test_label_list, gene_list


def get_symbol_index(gene_list, ncg_symbol, oncokb_unique):
    total_negative_index = [i for i in range(len(gene_list)) if
                            gene_list[i] not in ncg_symbol and gene_list[i] not in oncokb_unique]
    ncg_index = [i for i in range(len(gene_list)) if gene_list[i] in ncg_symbol]
    oncokb_unique_index = [i for i in range(len(gene_list)) if gene_list[i] in oncokb_unique]
    return oncokb_unique_index, ncg_index, total_negative_index, gene_list


def get_attention_adj(model, input_data):
    import networkx as nx
    graph = model.g
    total_attention = model.get_attention(input_data)
    src_nodes = graph.edges()[0]
    dst_nodes = graph.edges()[1]

    attention_df = pd.DataFrame(
        {'src': src_nodes.numpy(), 'dst': dst_nodes.numpy(), 'attention': total_attention.numpy()})

    g = nx.from_pandas_edgelist(attention_df, 'src', 'dst', ['attention'], create_using=nx.DiGraph)
    adj = nx.to_pandas_adjacency(g, weight='attention')
    adj = adj.sort_index()
    adj = adj.sort_index(axis=1)

    return adj


def get_cossim_list(normal_attention, tumor_attention):
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm
    tumor_cossim = list()
    for i in tqdm(range(len(normal_attention))):
        cossim = cosine_similarity([normal_attention.iloc[i]],
                                   [tumor_attention.iloc[i]])  # iloc[i]: inward, [i]: outward
        tumor_cossim.append(cossim[0][0])
    return tumor_cossim


def get_auroc_with_cv(model_list, input_list, subtype_list, gene_list, train_label_index_list, gene_index_list):
    normal_model = model_list['Normal']
    normal_data = input_list['Normal']
    normal_attention = get_attention_adj(normal_model, normal_data)

    train_label_list, train_index_list, test_label_list, test_index_list = train_label_index_list
    oncokb_unique_index, ncg_index, total_negative_index = gene_index_list

    cossim_dict = dict()
    for subtype in subtype_list:
        tumor_model = model_list[subtype]
        tumor_data = input_list[subtype]
        tumor_attention = get_attention_adj(tumor_model, tumor_data)
        cossim_list = get_cossim_list(normal_attention, tumor_attention)
        cossim_dict[subtype] = cossim_list

    cos_sim_df = pd.DataFrame(data=cossim_dict, index=gene_list)

    metric_result = get_auroc(cos_sim_df, subtype_list, train_label_list, train_index_list, test_label_list, test_index_list)
    pos_only_metric_result = get_auroc_pos_only(cos_sim_df, subtype_list, oncokb_unique_index, ncg_index, total_negative_index)

    return cos_sim_df, metric_result, pos_only_metric_result


def get_auroc(cos_sim_df, subtype_list, train_label_list, train_index_list, test_label_list, test_index_list):
    cos_sim_df.index.names = ['symbol']
    cos_sim_df = cos_sim_df.sort_index(axis=0)

    result = dict()
    for subtype in subtype_list:
        data = 1 - cos_sim_df[subtype]
        train_auprc = average_precision_score(train_label_list, data.iloc[train_index_list])
        train_auroc = roc_auc_score(train_label_list, data.iloc[train_index_list])
        test_auprc = average_precision_score(test_label_list, data.iloc[test_index_list])
        test_auroc = roc_auc_score(test_label_list, data.iloc[test_index_list])
        result[subtype] = [train_auprc, train_auroc, test_auprc, test_auroc]

    return result

def get_auroc_pos_only(cos_sim_df, subtype_list, oncokb_unique_index, ncg_index, total_negative_index):
    cos_sim_df.index.names = ['symbol']
    cos_sim_df = cos_sim_df.sort_index(axis=0)

    pos_only_train_index_list, pos_only_train_label_list, pos_only_test_index_list, pos_only_test_label_list = \
        get_pos_only_dataset(oncokb_unique_index, ncg_index, total_negative_index)

    result = dict()
    for subtype in subtype_list:
        data = 1 - cos_sim_df[subtype]

        train_auprc = average_precision_score(pos_only_train_label_list, data.iloc[pos_only_train_index_list])
        train_auroc = roc_auc_score(pos_only_train_label_list, data.iloc[pos_only_train_index_list])
        test_auprc = average_precision_score(pos_only_test_label_list, data.iloc[pos_only_test_index_list])
        test_auroc = roc_auc_score(pos_only_test_label_list, data.iloc[pos_only_test_index_list])
        result[subtype] = [train_auprc, train_auroc, test_auprc, test_auroc]

    return result
