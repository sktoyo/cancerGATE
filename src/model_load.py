from model.CancerGATE_old import *
from gatev2_utils import *
from preprocessing.Preprocess import load_preprocess_results

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle

import random
import numpy as np
import networkx as nx

import dgl
import tensorflow as tf


ncg_symbol, oncokb_unique, negative_symbol = get_symbol_list()
train_index_list, train_label_list, test_index_list, test_label_list, gene_list = get_index_list()
oncokb_unique_index, ncg_index, total_negative_index, gene_list = get_symbol_index(gene_list, ncg_symbol, oncokb_unique)

gpu_usage = False
# expression, mutation data in dataframe for each subtype

# edge and node information for network construction
input_dict = load_preprocess_results()
feature_dict = input_dict['subtype_x']
for subtype in feature_dict:
    feature_dict[subtype] = tf.convert_to_tensor(feature_dict[subtype], dtype=tf.float32)
humannet_edges_node1, humannet_edges_node2 = input_dict['edge_index']

# create network
humannet_dgl = dgl.graph((humannet_edges_node1, humannet_edges_node2))
if gpu_usage:
    humannet_dgl = humannet_dgl.to('/gpu:0')
else:
    pass

# case_name_list = ['Normal', "Her2", 'LumA', 'LumB', "Basal", "Tumor"]
case_name_list = ['Normal', "Tumor"]

# multi-modal
dim_hiddens = [128, 64, 32]  # calculated from performance check
num_layers = len(dim_hiddens) - 1
subtype_list = list(feature_dict.keys())
feat_drop = 0
attn_drop = 0
dropout = 0
negative_slope = 0.2
residual = True

heads = [8, 8]
activation = tf.keras.activations.relu
total_epoch = 150

model_name = 'GATE'

g = humannet_dgl
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)


# %%
def check_input_argue(input_argue, combine_input):
    if input_argue == 'mutation':
        in_dims = [mut_input.shape[1]]
        train_input = [mut_input]
    elif input_argue == 'expression':
        in_dims = [exp_input.shape[1]]
        train_input = [exp_input]
    elif input_argue == 'whole':
        in_dims = [exp_input.shape[1], mut_input.shape[1]]
        train_input = [exp_input, mut_input]
    elif input_argue == 'combine':
        in_dims = [combine_input.shape[1]]
        train_input = [combine_input]
    else:
        in_dims = None
        train_input = None
    return in_dims, train_input


input_argue = "combine"


def get_attention_adj(model, input_data, case_name):
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
    tumor_cossim = list()
    for i in tqdm(range(len(normal_attention))):
        cossim = cosine_similarity([normal_attention.iloc[i]],
                                   [tumor_attention.iloc[i]])  # iloc[i]: inward, [i]: outward
        tumor_cossim.append(cossim[0][0])
    return tumor_cossim


# %% md
# cosSim auroc
# %%
def get_auroc(cos_sim_df):
    cos_sim_df.index.names = ['symbol']
    cos_sim_df = cos_sim_df.sort_index(axis=0)

    result = dict()
    for subtype in subtype_list:
        data = 1 - cos_sim_df[subtype]

        train_auprc = average_precision_score(train_label_list, data.iloc[train_index_list])
        train_auroc = roc_auc_score(train_label_list, data.iloc[train_index_list])
        test_auprc = average_precision_score(test_label_list, data.iloc[test_index_list])
        test_auroc = roc_auc_score(test_label_list, data.iloc[test_index_list])
        # print(case, 'f1', f1_score(label, data))
        print(subtype, 'train auprc', train_auprc)
        print(subtype, 'train auroc', train_auroc)
        print(subtype, 'test auprc', test_auprc)
        print(subtype, 'test auroc', test_auroc)

        result[subtype] = [train_auprc, train_auroc, test_auprc, test_auroc]

    return result


def get_auroc_pos_only(cos_sim_df):
    cos_sim_df.index.names = ['symbol']
    cos_sim_df = cos_sim_df.sort_index(axis=0)

    pos_only_train_index_list, pos_only_train_label_list, pos_only_test_index_list, pos_only_test_label_list = \
        get_pos_only_dataset(oncoKB_unique_index, ncg_index, total_negative_index)

    result = dict()
    for subtype in subtype_list:
        data = 1 - cos_sim_df[subtype]

        train_auprc = average_precision_score(pos_only_train_label_list, data.iloc[pos_only_train_index_list])
        train_auroc = roc_auc_score(pos_only_train_label_list, data.iloc[pos_only_train_index_list])
        test_auprc = average_precision_score(pos_only_test_label_list, data.iloc[pos_only_test_index_list])
        test_auroc = roc_auc_score(pos_only_test_label_list, data.iloc[pos_only_test_index_list])
        # print(case, 'f1', f1_score(label, data))
        print(subtype, 'train auprc', train_auprc)
        print(subtype, 'train auroc', train_auroc)
        print(subtype, 'test auprc', test_auprc)
        print(subtype, 'test auroc', test_auroc)

        result[subtype] = [train_auprc, train_auroc, test_auprc, test_auroc]

    return result


# %%
def get_auroc_with_cv():
    normal_model = model_list['Normal']
    normal_data = input_list['Normal']
    normal_attention = get_attention_adj(normal_model, normal_data, 'Normal')

    cossim_dict = dict()
    # subtype_list = ['LumA', 'LumB', 'Basal', 'Her2', "Tumor"]
    subtype_list = ["Tumor"]
    for subtype in subtype_list:
        tumor_model = model_list[subtype]
        tumor_data = input_list[subtype]
        tumor_attention = get_attention_adj(tumor_model, tumor_data, subtype)
        cossim_list = get_cossim_list(normal_attention, tumor_attention)
        cossim_dict[subtype] = cossim_list

    cos_sim_df = pd.DataFrame(data=cossim_dict, index=gene_list)

    print(cv, subtype)
    metric_result = get_auroc(cos_sim_df)
    pos_only_metric_result = get_auroc_pos_only(cos_sim_df)

    return cos_sim_df, metric_result, pos_only_metric_result


# %%
def print_metric_average_std(total_metric):
    for key in total_metric.keys():
        print(key)
        cv_metric = total_metric[key]
        train_auprc_list = list()
        train_auroc_list = list()
        test_auprc_list = list()
        test_auroc_list = list()
        for cv in range(5):
            metric_result = cv_metric[cv]
            train_auprc, train_auroc, test_auprc, test_auroc = metric_result['Tumor']
            train_auprc_list.append(train_auprc)
            train_auroc_list.append(train_auroc)
            test_auprc_list.append(test_auprc)
            test_auroc_list.append(test_auroc)

        train_auprc_list = np.array(train_auprc_list)
        train_auroc_list = np.array(train_auroc_list)
        test_auprc_list = np.array(test_auprc_list)
        test_auroc_list = np.array(test_auroc_list)

        print('train auprc\tavg:{}\tstd:{}'.format(train_auprc_list.mean(), train_auprc_list.std()))
        print('train auroc\tavg:{}\tstd:{}'.format(train_auroc_list.mean(), train_auroc_list.std()))
        print('test auprc\tavg:{}\tstd:{}'.format(test_auprc_list.mean(), test_auprc_list.std()))
        print('test auroc\tavg:{}\tstd:{}'.format(test_auroc_list.mean(), test_auroc_list.std()))


# %%
total_metric = dict()
pos_only_total_metric = dict()
input_argue_list = ['combine']  # ['whole', 'mutation', 'expression']
subtype_list = ['Tumor']

for input_argue in input_argue_list:
    cv_metric = dict()
    pos_only_cv_metric = dict()
    for cv in range(5):
        print('cv:', str(cv), 'input_argue', input_argue)
        model_list = dict()
        input_list = dict()
        for case_name in case_name_list:
            combine_input = feature_dict[subtype]

            in_dims, train_input = check_input_argue(input_argue, exp_input, mut_input, combine_input)

            num_features = len(in_dims)
            model = CancerGATE(g, in_dims, dim_hiddens, heads, activation, feat_drop, attn_drop, negative_slope,
                               residual)

            checkpoint_path = "./checkpoints/CV{}_{}_{}_{}_{}_{}_full checkpoints/model.ckpt".format(cv, input_argue,
                                                                                                     model_name,
                                                                                                     case_name,
                                                                                                     total_epoch,
                                                                                                     ' '.join(
                                                                                                         [str(dim) for
                                                                                                          dim in
                                                                                                          dim_hiddens]))
            model.load_weights(checkpoint_path).expect_partial()

            input_list[case_name] = train_input
            model_list[case_name] = model

        structure_cossim_df, metric_result, pos_only_metric_result = get_auroc_with_cv()
        cv_metric[cv] = metric_result
        pos_only_cv_metric[cv] = pos_only_metric_result
    total_metric[input_argue] = cv_metric
    pos_only_total_metric[input_argue] = pos_only_cv_metric

    print_metric_average_std(total_metric)
    print()
    print('pos_only')
    print_metric_average_std(pos_only_total_metric)
    # %% md
    ## feature only models
    # %%
    total_metric = dict()
    pos_only_total_metric = dict()
    input_argue_list = ['whole']
    subtype_list = ['Tumor']

    for input_argue in input_argue_list:
        cv_metric = dict()
        pos_only_cv_metric = dict()
        for cv in range(10):
            print('cv:', str(cv), 'input_argue', input_argue)
            model_list = dict()
            input_list = dict()
            for case_name in case_name_list:
                exp_input = subtype_expression_input_dict[case_name]
                mut_input = subtype_mutation_input_dict[case_name]

                in_dims, train_input = check_input_argue(input_argue, exp_input, mut_input)

                num_features = len(in_dims)
                model = MultiModalAttentionGAE(g, in_dims, dim_hiddens, heads, activation, feat_drop, attn_drop,
                                               negative_slope,
                                               residual)

                checkpoint_path = "./checkpoints/CV{}_{}_{}_{}_{}_{}_feature_only checkpoints/model.ckpt".format(cv,
                                                                                                                 input_argue,
                                                                                                                 model_name,
                                                                                                                 case_name,
                                                                                                                 total_epoch,
                                                                                                                 ' '.join(
                                                                                                                     [
                                                                                                                         str(dim)
                                                                                                                         for
                                                                                                                         dim
                                                                                                                         in
                                                                                                                         dim_hiddens]))
                model.load_weights(checkpoint_path).expect_partial()

                input_list[case_name] = train_input
                model_list[case_name] = model

            structure_cossim_df, metric_result, pos_only_metric_result = get_auroc_with_cv()
            cv_metric[cv] = metric_result
            pos_only_cv_metric[cv] = pos_only_metric_result
        total_metric[input_argue] = cv_metric
        pos_only_total_metric[input_argue] = pos_only_cv_metric
    # %%
    print_metric_average_std(total_metric)
    print()
    print('pos_only')
    print_metric_average_std(pos_only_total_metric)
    # %% md
    ## structure only models
    # %%
    total_metric = dict()
    pos_only_total_metric = dict()
    input_argue_list = ['whole']
    subtype_list = ['Tumor']

    for input_argue in input_argue_list:
        cv_metric = dict()
        pos_only_cv_metric = dict()
        for cv in range(10):
            print('cv:', str(cv), 'input_argue', input_argue)
            model_list = dict()
            input_list = dict()
            for case_name in case_name_list:
                exp_input = subtype_expression_input_dict[case_name]
                mut_input = subtype_mutation_input_dict[case_name]

                in_dims, train_input = check_input_argue(input_argue, exp_input, mut_input)

                num_features = len(in_dims)
                model = MultiModalAttentionGAE(g, in_dims, dim_hiddens, heads, activation, feat_drop, attn_drop,
                                               negative_slope,
                                               residual)

                checkpoint_path = "./checkpoints/CV{}_{}_{}_{}_{}_{}_structure_only checkpoints/model.ckpt".format(cv,
                                                                                                                   input_argue,
                                                                                                                   model_name,
                                                                                                                   case_name,
                                                                                                                   total_epoch,
                                                                                                                   ' '.join(
                                                                                                                       [
                                                                                                                           str(dim)
                                                                                                                           for
                                                                                                                           dim
                                                                                                                           in
                                                                                                                           dim_hiddens]))
                model.load_weights(checkpoint_path).expect_partial()

                input_list[case_name] = train_input
                model_list[case_name] = model

            structure_cossim_df, metric_result, pos_only_metric_result = get_auroc_with_cv()
            cv_metric[cv] = metric_result
            pos_only_cv_metric[cv] = pos_only_metric_result
        total_metric[input_argue] = cv_metric
        pos_only_total_metric[input_argue] = pos_only_cv_metric
    # %%
    print_metric_average_std(total_metric)
    print()
    print('pos_only')
    print_metric_average_std(pos_only_total_metric)
    # %% md
    ## peturbation model - feature
    # %%
    with open("preprocessing/perturbed_networks.pickle", "rb") as f:
        perturbed_networks = pickle.load(f)

    with open("preprocessing/perturbed_features_tumor.pickle", "rb") as f:
        perturbed_tumor = pickle.load(f)

    with open("preprocessing/perturbed_features_normal.pickle", "rb") as f:
        perturbed_normal = pickle.load(f)
    # %%
    perturbed_features = list()
    for i in range(0, 5):
        perturbed = dict()
        perturbed['Tumor'] = perturbed_tumor[i]
        perturbed['Normal'] = perturbed_normal[i]
        perturbed_features.append(perturbed)

    # %%
    g = dgl.from_networkx(perturbed_networks[0])
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    perturbation_type = "feature"
    input_argue = 'whole'
    subtype_list = ['Tumor']

    print(perturbation_type)

    pert_metric = dict()
    pos_only_pert_metric = dict()
    for pert_percent in range(1, 5):
        cv_metric = dict()
        pos_only_cv_metric = dict()
        for cv in range(10):
            print('cv:', str(cv), 'pert_percent', str(25 * pert_percent))
            model_list = dict()
            input_list = dict()
            for case_name in case_name_list:
                exp_input = perturbed_features[pert_percent][case_name][0]
                mut_input = perturbed_features[pert_percent][case_name][1]

                in_dims, train_input = check_input_argue(input_argue, exp_input, mut_input)

                num_features = len(in_dims)
                model = MultiModalAttentionGAE(g, in_dims, dim_hiddens, heads, activation, feat_drop, attn_drop,
                                               negative_slope,
                                               residual)

                checkpoint_path = "./checkpoints/CV{}_{}_{}_{}_{}_{}_{}_{} checkpoints/model.ckpt".format(str(cv),
                                                                                                          input_argue,
                                                                                                          model_name,
                                                                                                          case_name,
                                                                                                          total_epoch,
                                                                                                          ' '.join(
                                                                                                              [str(dim)
                                                                                                               for dim
                                                                                                               in
                                                                                                               dim_hiddens]),
                                                                                                          perturbation_type,
                                                                                                          25 * pert_percent)
                model.load_weights(checkpoint_path).expect_partial()

                input_list[case_name] = train_input
                model_list[case_name] = model

            structure_cossim_df, metric_result, pos_only_metric_result = get_auroc_with_cv()
            cv_metric[cv] = metric_result
            pos_only_cv_metric[cv] = pos_only_metric_result
        pert_metric[pert_percent * 25] = cv_metric
        pos_only_pert_metric[pert_percent * 25] = pos_only_cv_metric
    # %%
    for pert_percent in range(1, 5):
        print("pert_percent", str(pert_percent * 25))
        cv_metric = pert_metric[pert_percent * 25]
        train_auprc_list = list()
        train_auroc_list = list()
        test_auprc_list = list()
        test_auroc_list = list()
        for cv in range(10):
            metric_result = cv_metric[cv]
            train_auprc, train_auroc, test_auprc, test_auroc = metric_result['Tumor']
            train_auprc_list.append(train_auprc)
            train_auroc_list.append(train_auroc)
            test_auprc_list.append(test_auprc)
            test_auroc_list.append(test_auroc)

        train_auprc_list = np.array(train_auprc_list)
        train_auroc_list = np.array(train_auroc_list)
        test_auprc_list = np.array(test_auprc_list)
        test_auroc_list = np.array(test_auroc_list)

        print('train auprc\tavg:{}\tstd:{}'.format(train_auprc_list.mean(), train_auprc_list.std()))
        print('train auroc\tavg:{}\tstd:{}'.format(train_auroc_list.mean(), train_auroc_list.std()))
        print('test auprc\tavg:{}\tstd:{}'.format(test_auprc_list.mean(), test_auprc_list.std()))
        print('test auroc\tavg:{}\tstd:{}'.format(test_auroc_list.mean(), test_auroc_list.std()))
    # %%
    print('pos_only')
    for pert_percent in range(1, 5):
        print("pert_percent", str(pert_percent * 25))
        cv_metric = pos_only_pert_metric[pert_percent * 25]
        train_auprc_list = list()
        train_auroc_list = list()
        test_auprc_list = list()
        test_auroc_list = list()
        for cv in range(10):
            metric_result = cv_metric[cv]
            train_auprc, train_auroc, test_auprc, test_auroc = metric_result['Tumor']
            train_auprc_list.append(train_auprc)
            train_auroc_list.append(train_auroc)
            test_auprc_list.append(test_auprc)
            test_auroc_list.append(test_auroc)

        train_auprc_list = np.array(train_auprc_list)
        train_auroc_list = np.array(train_auroc_list)
        test_auprc_list = np.array(test_auprc_list)
        test_auroc_list = np.array(test_auroc_list)

        print('train auprc\tavg:{}\tstd:{}'.format(train_auprc_list.mean(), train_auprc_list.std()))
        print('train auroc\tavg:{}\tstd:{}'.format(train_auroc_list.mean(), train_auroc_list.std()))
        print('test auprc\tavg:{}\tstd:{}'.format(test_auprc_list.mean(), test_auprc_list.std()))
        print('test auroc\tavg:{}\tstd:{}'.format(test_auroc_list.mean(), test_auroc_list.std()))
    # %% md
    ## peturbation model - network
    # %%
    perturbation_type = "network"
    input_argue = 'whole'
    subtype_list = ['Tumor']

    print(perturbation_type)
    pert_metric = dict()
    pos_only_pert_metric = dict()
    for pert_percent in range(1, 5):
        cv_metric = dict()
        pos_only_cv_metric = dict()
        g = dgl.from_networkx(perturbed_networks[pert_percent])
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        for cv in range(10):
            print('cv:', str(cv), 'pert_percent', str(25 * pert_percent))
            model_list = dict()
            input_list = dict()
            for case_name in case_name_list:
                exp_input = perturbed_features[0][case_name][0]
                mut_input = perturbed_features[0][case_name][1]

                in_dims, train_input = check_input_argue(input_argue, exp_input, mut_input)

                num_features = len(in_dims)
                model = MultiModalAttentionGAE(g, in_dims, dim_hiddens, heads, activation, feat_drop, attn_drop,
                                               negative_slope,
                                               residual)

                checkpoint_path = "./checkpoints/CV{}_{}_{}_{}_{}_{}_{}_{} checkpoints/model.ckpt".format(str(cv),
                                                                                                          input_argue,
                                                                                                          model_name,
                                                                                                          case_name,
                                                                                                          total_epoch,
                                                                                                          ' '.join(
                                                                                                              [str(dim)
                                                                                                               for dim
                                                                                                               in
                                                                                                               dim_hiddens]),
                                                                                                          perturbation_type,
                                                                                                          25 * pert_percent)
                model.load_weights(checkpoint_path).expect_partial()

                input_list[case_name] = train_input
                model_list[case_name] = model

            structure_cossim_df, metric_result, pos_only_metric_result = get_auroc_with_cv()
            cv_metric[cv] = metric_result
            pos_only_cv_metric[cv] = pos_only_metric_result
        pert_metric[pert_percent * 25] = cv_metric
        pos_only_pert_metric[pert_percent * 25] = pos_only_cv_metric
    # %%
    for pert_percent in range(1, 5):
        print("pert_percent", str(pert_percent * 25))
        cv_metric = pert_metric[pert_percent * 25]
        train_auprc_list = list()
        train_auroc_list = list()
        test_auprc_list = list()
        test_auroc_list = list()
        for cv in range(10):
            metric_result = cv_metric[cv]
            train_auprc, train_auroc, test_auprc, test_auroc = metric_result['Tumor']
            train_auprc_list.append(train_auprc)
            train_auroc_list.append(train_auroc)
            test_auprc_list.append(test_auprc)
            test_auroc_list.append(test_auroc)

        train_auprc_list = np.array(train_auprc_list)
        train_auroc_list = np.array(train_auroc_list)
        test_auprc_list = np.array(test_auprc_list)
        test_auroc_list = np.array(test_auroc_list)

        print('train auprc\tavg:{}\tstd:{}'.format(train_auprc_list.mean(), train_auprc_list.std()))
        print('train auroc\tavg:{}\tstd:{}'.format(train_auroc_list.mean(), train_auroc_list.std()))
        print('test auprc\tavg:{}\tstd:{}'.format(test_auprc_list.mean(), test_auprc_list.std()))
        print('test auroc\tavg:{}\tstd:{}'.format(test_auroc_list.mean(), test_auroc_list.std()))
    # %%
    print('pos_only')
    for pert_percent in range(1, 5):
        print("pert_percent", str(pert_percent * 25))
        cv_metric = pos_only_pert_metric[pert_percent * 25]
        train_auprc_list = list()
        train_auroc_list = list()
        test_auprc_list = list()
        test_auroc_list = list()
        for cv in range(10):
            metric_result = cv_metric[cv]
            train_auprc, train_auroc, test_auprc, test_auroc = metric_result['Tumor']
            train_auprc_list.append(train_auprc)
            train_auroc_list.append(train_auroc)
            test_auprc_list.append(test_auprc)
            test_auroc_list.append(test_auroc)

        train_auprc_list = np.array(train_auprc_list)
        train_auroc_list = np.array(train_auroc_list)
        test_auprc_list = np.array(test_auprc_list)
        test_auroc_list = np.array(test_auroc_list)

        print('train auprc\tavg:{}\tstd:{}'.format(train_auprc_list.mean(), train_auprc_list.std()))
        print('train auroc\tavg:{}\tstd:{}'.format(train_auroc_list.mean(), train_auroc_list.std()))
        print('test auprc\tavg:{}\tstd:{}'.format(test_auprc_list.mean(), test_auprc_list.std()))
        print('test auroc\tavg:{}\tstd:{}'.format(test_auroc_list.mean(), test_auroc_list.std()))
    # %% md
    ## peturbation model - both
    # %%
    perturbation_type = "both"
    input_argue = 'whole'
    subtype_list = ['Tumor']

    print(perturbation_type)
    pert_metric = dict()
    pos_only_pert_metric = dict()
    for pert_percent in range(1, 5):
        cv_metric = dict()
        pos_only_cv_metric = dict()
        g = dgl.from_networkx(perturbed_networks[pert_percent])
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        for cv in range(10):
            print('cv:', str(cv), 'pert_percent', str(25 * pert_percent))
            model_list = dict()
            input_list = dict()
            for case_name in case_name_list:
                exp_input = perturbed_features[pert_percent][case_name][0]
                mut_input = perturbed_features[pert_percent][case_name][1]

                in_dims, train_input = check_input_argue(input_argue, exp_input, mut_input)

                num_features = len(in_dims)
                model = MultiModalAttentionGAE(g, in_dims, dim_hiddens, heads, activation, feat_drop, attn_drop,
                                               negative_slope,
                                               residual)

                checkpoint_path = "./checkpoints/CV{}_{}_{}_{}_{}_{}_{}_{} checkpoints/model.ckpt".format(str(cv),
                                                                                                          input_argue,
                                                                                                          model_name,
                                                                                                          case_name,
                                                                                                          total_epoch,
                                                                                                          ' '.join(
                                                                                                              [str(dim)
                                                                                                               for dim
                                                                                                               in
                                                                                                               dim_hiddens]),
                                                                                                          perturbation_type,
                                                                                                          25 * pert_percent)
                model.load_weights(checkpoint_path).expect_partial()

                input_list[case_name] = train_input
                model_list[case_name] = model

            structure_cossim_df, metric_result, pos_only_metric_result = get_auroc_with_cv()
            cv_metric[cv] = metric_result
            pos_only_cv_metric[cv] = pos_only_metric_result
        pert_metric[pert_percent * 25] = cv_metric
        pos_only_pert_metric[pert_percent * 25] = pos_only_cv_metric
    # %%
    for pert_percent in range(1, 5):
        print("pert_percent", str(pert_percent * 25))
        cv_metric = pert_metric[pert_percent * 25]
        train_auprc_list = list()
        train_auroc_list = list()
        test_auprc_list = list()
        test_auroc_list = list()
        for cv in range(10):
            metric_result = cv_metric[cv]
            train_auprc, train_auroc, test_auprc, test_auroc = metric_result['Tumor']
            train_auprc_list.append(train_auprc)
            train_auroc_list.append(train_auroc)
            test_auprc_list.append(test_auprc)
            test_auroc_list.append(test_auroc)

        train_auprc_list = np.array(train_auprc_list)
        train_auroc_list = np.array(train_auroc_list)
        test_auprc_list = np.array(test_auprc_list)
        test_auroc_list = np.array(test_auroc_list)

        print('train auprc\tavg:{}\tstd:{}'.format(train_auprc_list.mean(), train_auprc_list.std()))
        print('train auroc\tavg:{}\tstd:{}'.format(train_auroc_list.mean(), train_auroc_list.std()))
        print('test auprc\tavg:{}\tstd:{}'.format(test_auprc_list.mean(), test_auprc_list.std()))
        print('test auroc\tavg:{}\tstd:{}'.format(test_auroc_list.mean(), test_auroc_list.std()))
    # %%
    print('pos_only')
    for pert_percent in range(1, 5):
        print("pert_percent", str(pert_percent * 25))
        cv_metric = pos_only_pert_metric[pert_percent * 25]
        train_auprc_list = list()
        train_auroc_list = list()
        test_auprc_list = list()
        test_auroc_list = list()
        for cv in range(10):
            metric_result = cv_metric[cv]
            train_auprc, train_auroc, test_auprc, test_auroc = metric_result['Tumor']
            train_auprc_list.append(train_auprc)
            train_auroc_list.append(train_auroc)
            test_auprc_list.append(test_auprc)
            test_auroc_list.append(test_auroc)

        train_auprc_list = np.array(train_auprc_list)
        train_auroc_list = np.array(train_auroc_list)
        test_auprc_list = np.array(test_auprc_list)
        test_auroc_list = np.array(test_auroc_list)

        print('train auprc\tavg:{}\tstd:{}'.format(train_auprc_list.mean(), train_auprc_list.std()))
        print('train auroc\tavg:{}\tstd:{}'.format(train_auroc_list.mean(), train_auroc_list.std()))
        print('test auprc\tavg:{}\tstd:{}'.format(test_auprc_list.mean(), test_auprc_list.std()))
        print('test auroc\tavg:{}\tstd:{}'.format(test_auroc_list.mean(), test_auroc_list.std()))
    # %% md
    ## perturbation model - power-law network
    # %%
    g = dgl.from_networkx(perturbed_networks[-1])
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    perturbation_type_list = ["network-random_network", "both-random_network"]
    input_argue = 'whole'
    subtype_list = ['Tumor']
    pert_percent = 4

    result = dict()
    for perturbation_type in perturbation_type_list:
        print(perturbation_type)
        pert_metric = dict()
        pos_only_pert_metric = dict()

        cv_metric = dict()
        pos_only_cv_metric = dict()

        for cv in range(10):
            print('cv:', str(cv), 'pert_percent', str(25 * pert_percent))
            model_list = dict()
            input_list = dict()
            for case_name in case_name_list:

                if perturbation_type == "network-random_network":
                    exp_input = perturbed_features[0][case_name][0]
                    mut_input = perturbed_features[0][case_name][1]
                else:
                    exp_input = perturbed_features[pert_percent][case_name][0]
                    mut_input = perturbed_features[pert_percent][case_name][1]
                in_dims, train_input = check_input_argue(input_argue, exp_input, mut_input)

                num_features = len(in_dims)
                model = MultiModalAttentionGAE(g, in_dims, dim_hiddens, heads, activation, feat_drop, attn_drop,
                                               negative_slope,
                                               residual)

                checkpoint_path = "./checkpoints/CV{}_{}_{}_{}_{}_{}_{}_{} checkpoints/model.ckpt".format(str(cv),
                                                                                                          input_argue,
                                                                                                          model_name,
                                                                                                          case_name,
                                                                                                          total_epoch,
                                                                                                          ' '.join(
                                                                                                              [str(dim)
                                                                                                               for dim
                                                                                                               in
                                                                                                               dim_hiddens]),
                                                                                                          perturbation_type,
                                                                                                          25 * pert_percent)
                model.load_weights(checkpoint_path).expect_partial()

                input_list[case_name] = train_input
                model_list[case_name] = model

            structure_cossim_df, metric_result, pos_only_metric_result = get_auroc_with_cv()
            cv_metric[cv] = metric_result
            pos_only_cv_metric[cv] = pos_only_metric_result
        pert_metric[pert_percent * 25] = cv_metric
        pos_only_pert_metric[pert_percent * 25] = pos_only_cv_metric

        result[perturbation_type] = [pert_metric, pos_only_pert_metric]
    # %%
    for perturbation_type in perturbation_type_list:
        pert_metric = result[perturbation_type][0]
        print(perturbation_type)
        cv_metric = pert_metric[pert_percent * 25]
        train_auprc_list = list()
        train_auroc_list = list()
        test_auprc_list = list()
        test_auroc_list = list()
        for cv in range(10):
            metric_result = cv_metric[cv]
            train_auprc, train_auroc, test_auprc, test_auroc = metric_result['Tumor']
            train_auprc_list.append(train_auprc)
            train_auroc_list.append(train_auroc)
            test_auprc_list.append(test_auprc)
            test_auroc_list.append(test_auroc)

        train_auprc_list = np.array(train_auprc_list)
        train_auroc_list = np.array(train_auroc_list)
        test_auprc_list = np.array(test_auprc_list)
        test_auroc_list = np.array(test_auroc_list)

        print('train auprc\tavg:{}\tstd:{}'.format(train_auprc_list.mean(), train_auprc_list.std()))
        print('train auroc\tavg:{}\tstd:{}'.format(train_auroc_list.mean(), train_auroc_list.std()))
        print('test auprc\tavg:{}\tstd:{}'.format(test_auprc_list.mean(), test_auprc_list.std()))
        print('test auroc\tavg:{}\tstd:{}'.format(test_auroc_list.mean(), test_auroc_list.std()))
    # %%
    print("pos only")
    for perturbation_type in perturbation_type_list:
        pert_metric = result[perturbation_type][1]
        print(perturbation_type)
        cv_metric = pert_metric[pert_percent * 25]
        train_auprc_list = list()
        train_auroc_list = list()
        test_auprc_list = list()
        test_auroc_list = list()
        for cv in range(10):
            metric_result = cv_metric[cv]
            train_auprc, train_auroc, test_auprc, test_auroc = metric_result['Tumor']
            train_auprc_list.append(train_auprc)
            train_auroc_list.append(train_auroc)
            test_auprc_list.append(test_auprc)
            test_auroc_list.append(test_auroc)

        train_auprc_list = np.array(train_auprc_list)
        train_auroc_list = np.array(train_auroc_list)
        test_auprc_list = np.array(test_auprc_list)
        test_auroc_list = np.array(test_auroc_list)

        print('train auprc\tavg:{}\tstd:{}'.format(train_auprc_list.mean(), train_auprc_list.std()))
        print('train auroc\tavg:{}\tstd:{}'.format(train_auroc_list.mean(), train_auroc_list.std()))
        print('test auprc\tavg:{}\tstd:{}'.format(test_auprc_list.mean(), test_auprc_list.std()))
        print('test auroc\tavg:{}\tstd:{}'.format(test_auroc_list.mean(), test_auroc_list.std()))
