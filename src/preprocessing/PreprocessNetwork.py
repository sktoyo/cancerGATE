import pandas as pd


class PreprocessNetwork:
    def __init__(self, network_name, experiment_genes):
        self.network_path = "../../data/network/"
        self.network_name = network_name
        self.experiment_genes = experiment_genes

    def preprocess(self):
        """
        Get edge list from the network
        return: edge list in [[node1, node2]] format
        """
        if self.check_network_edges():
            return self.load_network_edges()
        else:
            if self.network_name == "humannet_edges_FN":
                network_edges = self.preprocess_humannet_edges()
                self.save_network_edges(network_edges)
                return network_edges

    def check_network_edges(self):
        """
        Check whether preprocessed PPI exists or not

        return 'bool'
        """
        import os
        if os.path.exists(self.network_path + "{}.pickle".format(self.network_name)):
            return True
        else:
            return False

    def load_network_edges(self):
        """
        load network edges from pickle file
        return:  edge list in [[node1, node2]] format
        """
        import pickle
        with open(self.network_path + "{}.pickle".format(self.network_name), 'rb') as f:
            network_edges = pickle.load(f)
            return network_edges

    def preprocess_humannet_edges(self):
        entrez_id_dict = self.get_entrez_id()

        humannet_df = self.load_humannet_original_edges()
        humannet_df = self.fix_entrez_ids_humannet(humannet_df)
        humannet_edges = self.get_humannet_edges(humannet_df, entrez_id_dict)
        humannet_edges = self.refine_network(humannet_edges)
        return humannet_edges

    def get_entrez_id(self):
        """
        Get dictionary of entrezid and symbol

        return 'dict: {entrezid:symbol}
        """
        gene_id_info = pd.read_csv(self.network_path + 'Homo_sapiens.gene_info.txt', sep='\t')
        gene_id_info = gene_id_info.drop(columns=['#tax_id'])
        gene_id_info = gene_id_info[['GeneID', 'Symbol']]

        entrez_id_list = gene_id_info['GeneID'].tolist()
        entrez_symbol = gene_id_info['Symbol'].tolist()
        entrez_id_dict = dict()
        for i, entrez_id in enumerate(entrez_id_list):
            entrez_id_dict[str(entrez_id)] = entrez_symbol[i]
        return entrez_id_dict

    def load_humannet_original_edges(self):
        """
        Get original edge dataframe of humannet

        return 'DataFrame: edge datafame
        """
        humannet_df = pd.read_csv(self.network_path + 'HumanNet-FN.tsv', sep='\t')
        humannet_df = humannet_df.astype('str')
        return humannet_df

    @staticmethod
    def fix_entrez_ids_humannet(humannet_df):
        """
        Fix entrez ID errors in edge dataframe

        return:
        """
        humannet_df = humannet_df[humannet_df.EntrezGeneID1 != '10896']
        humannet_df = humannet_df[humannet_df.EntrezGeneID2 != '10896']

        humannet_df = humannet_df[humannet_df.EntrezGeneID1 != '285464']
        humannet_df = humannet_df[humannet_df.EntrezGeneID2 != '285464']

        humannet_df = humannet_df[humannet_df.EntrezGeneID1 != '729574']
        humannet_df = humannet_df[humannet_df.EntrezGeneID2 != '729574']

        humannet_df = humannet_df[humannet_df.EntrezGeneID1 != '10638']
        humannet_df = humannet_df[humannet_df.EntrezGeneID2 != '10638']

        humannet_df = humannet_df.replace({'26148': '84458'})
        humannet_df = humannet_df.replace({'23285': '284697'})
        humannet_df = humannet_df.replace({'114299': '445815'})
        humannet_df = humannet_df.replace({'117153': '4253'})
        humannet_df = humannet_df.replace({'11217': '445815'})
        humannet_df = humannet_df.replace({'338809': '440107'})
        return humannet_df

    def get_humannet_edges(self, humannet_df, entrez_id_dict):
        """
        Get list of edges from dataframe format

        param humannet_df:

        param entrez_id_dict:

        return:
        """
        from tqdm import tqdm
        humannet_edges = list()
        for index, row in tqdm(humannet_df.iterrows()):
            gene1 = entrez_id_dict[row['EntrezGeneID1']]
            gene2 = entrez_id_dict[row['EntrezGeneID2']]
            if gene1 in self.experiment_genes and gene2 in self.experiment_genes:
                humannet_edges.append([gene1, gene2])
        return humannet_edges

    @staticmethod
    def refine_network(network_edges):
        """
        Remove self loop, remove duplicated reverse edges, make bidirectionality

        return: refined network_edges
        """
        import itertools
        network_edges = list(humannet_edges for humannet_edges, _ in itertools.groupby(network_edges))
        network_edges = [edge for edge in network_edges if edge[0] != edge[1]]
        network_edges_reverse = [[edge[1], edge[0]] for edge in network_edges]
        network_edges = network_edges + network_edges_reverse
        network_edges = list(set(map(tuple, network_edges)))
        return network_edges

    def save_network_edges(self, network_edges):
        import pickle
        with open(self.network_path + "{}.pickle".format(self.network_name), 'wb') as f:
            pickle.dump(network_edges, f)
        return 0
