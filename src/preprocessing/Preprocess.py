from PreprocessExpression import *
from PreprocessMutation import *
from PreprocessNetwork import *


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
    print(len(network_edges))

    subtype_exp_dict, subtype_mut_dict = filter_experiment_genes(subtype_exp_dict, subtype_mut_dict, network_edges)

    gene_index_dict = get_save_gene_index(subtype_exp_dict)


    # # change symbol to gene index in gene_index_dict
    humannet_edges = [[gene_index_dict[edge[0]], gene_index_dict[edge[1]]] for edge in humannet_edges]

    humannet_edges_node1 = [edge[0] for edge in humannet_edges]
    humannet_edges_node2 = [edge[1] for edge in humannet_edges]

    ##############
    def min_max_normalization(df):
        if sum(df.max() - df.min()) != 0:
            result = (df - df.min()) / (df.max() - df.min())
        else:
            result = df
        return result

    orig_subtype_dict = getSubtypeDict()
    subtype_dict = dict()
    for key, value in orig_subtype_dict.items():
        if value in subtype_dict.keys():
            subtype_dict[value].append(key)
        else:
            subtype_dict[value] = [key]

    subtype_list = [subtype for subtype in subtype_dict.keys()]
    subtype_list.append('Tumor')

    with open(preprocess_dir + 'subtypes_exp_dict_raw.pickle', 'wb') as f:
        pickle.dump(subtype_expression_dict, f)

    ##############
    for subtype in subtype_list:
        subtype_expression_dict[subtype] = min_max_normalization(subtype_expression_dict[subtype])
        subtype_mutation_dict[subtype] = min_max_normalization(subtype_mutation_dict[subtype])

    if average:
        with open(preprocess_dir + 'subtypes_exp_dict.pickle', 'wb') as f:
            pickle.dump(subtype_expression_dict, f)
    else:
        with open(preprocess_dir + 'subtypes_exp_dict_ori.pickle', 'wb') as f:
            pickle.dump(subtype_expression_dict, f)

    with open(preprocess_dir + 'subtypes_mutation_dict.pickle', 'wb') as f:
        pickle.dump(subtype_mutation_dict, f)
    with open(preprocess_dir + 'humannet_node1_node2.pickle', 'wb') as f:
        pickle.dump([humannet_edges_node1, humannet_edges_node2], f)


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


def filter_expression_genes(subtype_exp_dict, intersection_genes):
    for subtype in subtype_exp_dict:
        subtype_exp_dict[subtype] = subtype_exp_dict[subtype][
            subtype_exp_dict[subtype].index.isin(intersection_genes)]
    return subtype_exp_dict


def filter_mutation_genes(subtype_mut_dict, intersection_genes):
    for subtype in subtype_mut_dict:
        subtype_mut_dict[subtype] = subtype_mut_dict[subtype][
            subtype_mut_dict[subtype].index.isin(intersection_genes)]
    return subtype_mut_dict


def filter_experiment_genes(subtype_exp_dict, subtype_mut_dict, network_edges):

    intersection_genes = get_intersection_genes(subtype_exp_dict, subtype_mut_dict, network_edges)
    subtype_exp_dict = filter_expression_genes(subtype_exp_dict, intersection_genes)
    subtype_mut_dict = filter_mutation_genes(subtype_mut_dict, intersection_genes)
    return subtype_exp_dict, subtype_mut_dict


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
    exp_gene_list = subtype_exp_dict.index.tolist()
    for i in range(len(exp_gene_list)):
        gene_info = [str(i), exp_gene_list[i]]
        gene_index_dict[exp_gene_list[i]] = i
    return gene_index_dict

preprocess()

