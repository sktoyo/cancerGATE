import pandas as pd
import os


class PreprocessExpression:
    def __init__(self, tissue, cancer_type):
        self.expression_path = "../../data/expression/"
        self.mutation_path = "../../data/mutation/"
        self.network_path = "../../data/network/"
        self.tissue = tissue
        self.cancer_type = cancer_type

    def preprocess(self):
        """
        Preprocess the TCGA TPM data and GTEx TPM data to average data

        return 'dict': expression dict with dataframe format
        """

        gtex_exp_df = self.get_gtex_exp_df()
        gtex_exp_df = self.filter_sparse_expression(gtex_exp_df)

        tcga_exp_df = self.get_TCGA_exp_df()
        tcga_exp_df = self.filter_sparse_expression(tcga_exp_df)
        tcga_exp_df = self.preprocess_sample_barcode(tcga_exp_df)
        tcga_exp_df = self.filter_tcga_samples(tcga_exp_df)

        gtex_exp_df, tcga_exp_df = self.get_index_intersection(gtex_exp_df, tcga_exp_df)
        subtype_exp_dict = self.get_subtype_exp_dict(gtex_exp_df, tcga_exp_df)
        subtype_exp_dict = self.get_average_exp(subtype_exp_dict)
        return subtype_exp_dict

    def preprocess_sample_barcode(self, tcga_exp_df):
        columns = tcga_exp_df.columns.to_list()
        preprocessed_columns = list()
        for column in columns:
            new_column = '-'.join(column.split('-')[:4])
            preprocessed_columns.append(new_column)
        tcga_exp_df.columns = preprocessed_columns
        return tcga_exp_df

    @staticmethod
    def get_average_exp(subtype_exp_dict):
        """
        Make average information of expressions

        return:
        """
        for subtype in subtype_exp_dict:
            avg_series = subtype_exp_dict[subtype].mean(axis=1)
            subtype_exp_dict[subtype] = pd.DataFrame(avg_series, columns=['avg_TPM'])
        return subtype_exp_dict

    def get_gtex_exp_df(self):
        """
        Get the GTEx TPM data, filter related tissue only dataframe

        return 'dict': expression dict with dataframe format
        """
        gtex_exp_df = pd.read_csv(self.expression_path + "{}-rsem-fpkm-gtex.txt".format(self.tissue),
                                  sep='\t', index_col=0)
        gtex_exp_df.drop(columns=['Entrez_Gene_Id'], inplace=True)
        return gtex_exp_df

        # if self.check_gtex_file():
        #     gtex_exp_df = pd.read_csv(self.expression_path + "gtex_{}_exp.tsv".format(self.tissue),
        #                               sep='\t', index_col=0)
        # else:
        #     gtex_exp_df = self.get_tissue_gtex()
        # return gtex_exp_df

    def get_TCGA_exp_df(self):
        """
        Get the GTEx TPM data, filter related tissue only dataframe

        return 'dict': expression dict with dataframe format
        """
        tcga_exp_df = pd.read_csv(self.expression_path + "{}-rsem-fpkm-tcga-t.txt".format(self.cancer_type),
                                  sep='\t', index_col=0)
        tcga_exp_df.drop(columns=['Entrez_Gene_Id'], inplace=True)
        return tcga_exp_df

    # def check_gtex_file(self):
    #     """
    #     Check whether file exists
    #     return: 'bool'
    #     """
    #     if os.path.exists(self.expression_path + "gtex_{}_exp.tsv".format(self.tissue)):
    #         return True
    #     else:
    #         return False

    # def get_tissue_gtex(self):
    #     """
    #     Filter gtex expression with tissue
    #
    #     return:
    #     """
    #     gtex_df = self.get_tpm_df_gtex()
    #     gtex_label_info = self.get_gtex_label_info()
    #
    #     gtex_sample_list = gtex_label_info[gtex_label_info['SMTS'] == self.tissue].index.to_list()
    #     gtex_sample_list_filtered = [sample for sample in gtex_sample_list if sample in gtex_df.columns.tolist()]
    #     gtex_sample_list_filtered = ['Description'] + gtex_sample_list_filtered
    #     gtex_exp_df = gtex_df[gtex_sample_list_filtered]
    #     gtex_exp_df = self.filter_sparse_expression(gtex_exp_df)
    #     gtex_exp_df.to_csv(self.expression_path + "gtex_{}_exp.tsv".format(self.tissue), sep="\t")
    #     return gtex_exp_df

    # def get_tpm_df_gtex(self):
    #     """
    #     Get all gtex tpm datafrme
    #
    #     return: gtex tpm dataframe
    #     """
    #     expression_df = pd.read_csv(self.expression_path + "GTEx_tpm.gct", sep='\t', index_col=0, skiprows=2)
    #     return expression_df

    # def get_gtex_label_info(self):
    #     """
    #     Get gtex tissue label information
    #
    #     return: gtex label information
    #     """
    #     gtex_label_info = pd.read_csv(self.expression_path + 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt',
    #                                   sep='\t', index_col=0)
    #     return gtex_label_info

    # def get_TCGA_tpm_df(self):
    #     """
    #     Get all gtex tpm datafrme
    #
    #     return: gtex tpm dataframe
    #     """
    #     expression_df = pd.read_csv(self.expression_path + 'TCGA_{}_TPM_symbol.tsv'.format(self.cancer_type),
    #                                 sep='\t', index_col=0)
    #     return expression_df

    def get_subtype_dict(self):
        """
        Get subtype information of TCGA BRCA

        return: {subtype:sample id} dictionary
        """
        from collections import defaultdict
        file_info_df = pd.read_csv(self.expression_path + "TCGA_{}_mol_subtype.tsv".format(self.cancer_type), sep='\t')
        subtype_dict = defaultdict(list)

        for index, row in file_info_df.iterrows():
            subtype = row['Subtype_mRNA']
            sample_id = row['pan.samplesID']
            sample_id = '-'.join(sample_id.split('-')[:4])
            subtype_dict[subtype].append(sample_id)
        return subtype_dict

    @staticmethod
    def filter_sparse_expression(exp_df):
        """
        filter gene have expression at least 20% of samples

        return: filtered gtex expression dataframe
        """
        exp_df = exp_df[exp_df.astype('bool').mean(axis=1) >= 0.2]
        exp_df = exp_df.groupby('Hugo_Symbol').mean()
        return exp_df

    @staticmethod
    def get_index_intersection(gtex_exp_df, tcga_exp_df):
        """
        Make dataframes have intersection index

        param 'gtex_exp_df'

        param 'tcga_exp_df'

        :return gtex_exp_df, tcga_exp_df
        """
        intersection_gene = tcga_exp_df.index.intersection(gtex_exp_df.index)
        tcga_exp_df = tcga_exp_df.loc[intersection_gene]
        tcga_exp_df = tcga_exp_df.sort_index()

        gtex_exp_df = gtex_exp_df.loc[intersection_gene]
        gtex_exp_df = gtex_exp_df.sort_index()

        return gtex_exp_df, tcga_exp_df

    def filter_tcga_samples(self, tcga_exp_df):
        """
        Exclude tcga samples that does not have mutation information

        return
        """
        sample_list = self.get_sample_list()
        tcga_exp_df = tcga_exp_df.loc[:, tcga_exp_df.columns.isin(sample_list)]
        return tcga_exp_df

    def get_sample_list(self):
        """
        Get intersection of samples between expression and mutation of TCGA

        return: sample list
        """
        mutation_list = pd.read_csv(self.mutation_path + "mutation_sample_list.tsv", sep="\t", index_col=False)
        mutation_list = mutation_list['Mutation Sample'].to_list()
        return mutation_list
        # import pickle
        # with open(self.expression_path + 'mut_expression_intersection_list.pkl', 'rb') as f:
        #     return pickle.load(f)

    def get_subtype_exp_dict(self, gtex_exp_df, tcga_exp_df):
        subtype_dict = self.get_subtype_dict()
        subtype_exp_dict = dict()
        for subtype in subtype_dict.keys():
            subtype_expression = self.isolate_sample(tcga_exp_df, subtype_dict[subtype])
            subtype_exp_dict[subtype] = subtype_expression

        subtype_exp_dict['Normal'] = gtex_exp_df  # Change TCGA normal -> GTEx normal

        tumor_expression = None
        for subtype in subtype_exp_dict.keys():
            if subtype == 'Normal':
                continue
            elif tumor_expression is None:
                tumor_expression = subtype_exp_dict[subtype]
            else:
                tumor_expression = pd.concat([tumor_expression, subtype_exp_dict[subtype]], axis=1)

        subtype_exp_dict['Tumor'] = tumor_expression
        return subtype_exp_dict

    @staticmethod
    def isolate_sample(orig_df, sample_list):
        result = orig_df[orig_df.columns.intersection(sample_list)]
        return result
