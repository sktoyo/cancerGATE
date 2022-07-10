from preprocessing.PreprocessExpression import *
from preprocessing.PreprocessMutation import *
from preprocessing.PreprocessNetwork import *


def preprocess():
    """
    Save feature matrix and network pickle
    """
    pre_exp = PreprocessExpression('breast', 'brca')
    subtype_exp_dict = pre_exp.preprocess()

    pre_mut = PreprocessMutation('brca')
    subtype_mut_dict = pre_mut.preprocess()

    experiment_genes = get_experiment_genes(subtype_exp_dict, subtype_mut_dict)
    pre_net = PreprocessNetwork('humannet_edges_FN', experiment_genes)
    network_edges = pre_net.preprocess()

    subtype_exp_dict, subtype_mut_dict = filter_experiment_genes(subtype_exp_dict, subtype_mut_dict, network_edges)

    gene_index_dict = get_save_gene_index(subtype_exp_dict)
    network_edges = convert_symbol2index(network_edges, gene_index_dict)
    network_edges = transpose_network_edges(network_edges)

    subtype_experiment_dict = concat_exp_mut(subtype_exp_dict, subtype_mut_dict)
    save_preprocess_results(subtype_experiment_dict, network_edges)
    return subtype_experiment_dict


def get_experiment_genes(subtype_exp_dict, subtype_mut_dict):
    """
    Get intersection of genes between expression, mutation

    param subtype_exp_dict

    param subtype_mut_dict

    return: intersection of genes between expression, mutation
    """
    expression_gene_list = subtype_exp_dict['Tumor'].index.to_list()
    mutation_gene_list = subtype_mut_dict['Tumor'].index.to_list()
    experiment_genes = list(set(expression_gene_list) & set(mutation_gene_list))
    return experiment_genes


def filter_experiment_genes(subtype_exp_dict, subtype_mut_dict, network_edges):
    """
    Filter the gene in expression and mutation dataframe by intersection of expression, network, mutation

    param subtype_exp_dict:

    param subtype_mut_dict:

    param network_edges:

    return: filtered expression data and mutation data
    """
    intersection_genes = get_intersection_genes(subtype_exp_dict, subtype_mut_dict, network_edges)
    subtype_exp_dict = filter_gene_index(subtype_exp_dict, intersection_genes)
    subtype_mut_dict = filter_gene_index(subtype_mut_dict, intersection_genes)
    return subtype_exp_dict, subtype_mut_dict


def get_intersection_genes(subtype_exp_dict, subtype_mut_dict, network_edges):
    """
    Get intersection of genes between expression, mutation, network

    param subtype_exp_dict

    param subtype_mut_dict

    param network_edges

    return: intersection of genes between expression, mutation, network
    """
    expression_gene_list = subtype_exp_dict['Tumor'].index.to_list()
    mutation_gene_list = subtype_mut_dict['Tumor'].index.to_list()
    network_genes = set([edge[0] for edge in network_edges])
    intersection_genes = list(set(expression_gene_list) & set(mutation_gene_list) & set(network_genes))
    return intersection_genes


def filter_gene_index(subtype_dict, intersection_genes):
    """
    Filter the gene in expression dataframe by intersection of expression, network, mutation
    :param subtype_dict:
    :param intersection_genes:
    :return:
    """
    for subtype in subtype_dict:
        subtype_dict[subtype] = subtype_dict[subtype][
            subtype_dict[subtype].index.isin(intersection_genes)]
    return subtype_dict


def get_save_gene_index(subtype_exp_dict):
    """
    Save gene index in tsv file

    param subtype_exp_dict:

    return: gene_index_dict
    """
    import pandas as pd
    gene_index_dict = get_gene_index(subtype_exp_dict)
    gene_index = pd.Series(gene_index_dict, name='index')
    gene_index_df = pd.DataFrame(gene_index)
    gene_index_df.to_csv('../../data/gene_index.tsv', sep='\t')
    return gene_index_dict


def get_gene_index(subtype_exp_dict):
    """
    Get gene index from experiment list

    return
    """
    gene_index_dict = dict()
    exp_gene_list = subtype_exp_dict['Tumor'].index.tolist()
    for i in range(len(exp_gene_list)):
        gene_index_dict[exp_gene_list[i]] = i
    return gene_index_dict


def convert_symbol2index(network_edges, gene_index_dict):
    """
    Change symbol to gene index in gene_index_dict

    param network_edges:

    param gene_index_dict:

    return: network_edges in index
    """
    network_edges = [[gene_index_dict[edge[0]], gene_index_dict[edge[1]]] for edge in network_edges]
    return network_edges


def transpose_network_edges(network_edges):
    """
    Transpose the shape of network edges (num_edges, 2) -> (2, num_edges)
    param network_edges:
    return: transposed network_edges
    """

    network_node1 = [edge[0] for edge in network_edges]
    network_node2 = [edge[1] for edge in network_edges]
    return [network_node1, network_node2]


def min_max_normalization(df):
    if sum(df.max() - df.min()) != 0:
        result = (df - df.min()) / (df.max() - df.min())
    else:
        result = df
    return result


def concat_exp_mut(subtype_exp_dict, subtype_mut_dict):
    """
    Concatenate expression data and mutation data
    :return: subtype_experiment_dict
    """
    import pandas as pd
    subtype_experiment_dict = dict()
    for subtype in subtype_exp_dict:
        exp_df = subtype_exp_dict[subtype]
        exp_df = min_max_normalization(exp_df)
        mut_df = subtype_mut_dict[subtype]
        if not subtype == 'Normal':
            mut_df = min_max_normalization(mut_df)
        concat_df = pd.concat([exp_df, mut_df], axis=1)
        subtype_experiment_dict[subtype] = concat_df

    return subtype_experiment_dict


def save_preprocess_results(subtype_experiment_dict, network_edges):
    """
    Save preprocessing results
    :param subtype_experiment_dict:
    :param network_edges:
    :return: 0
    """
    preprocess_results = dict()
    preprocess_results['edge_index'] = network_edges
    preprocess_results['subtype_x'] = subtype_experiment_dict
    import pickle
    with open("../../data/" + 'input_data.pkl', 'wb') as f:
        pickle.dump(preprocess_results, f)
    return 0


def load_preprocess_results():
    """
    lod preprocessing result
    return: 'dict' of edge_index and expression data
    """
    import pickle
    with open("../data/input_data.pkl", 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    subtype_experiment_dict = preprocess()
