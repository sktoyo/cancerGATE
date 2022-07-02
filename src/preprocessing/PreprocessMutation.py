import pandas as pd


class PreprocessMutation:
    def __init__(self, cancer_type):
        self.expression_path = "../../data/expression/"
        self.mutation_path = "../../data/mutation/"
        self.cancer_type = cancer_type

    def preprocess(self):
        """
        Preprocess the TCGA mutation data

        return 'dict': mutation dict with dataframe format
        """
        subtype_dict = self.get_subtype_dict()
        subtype_list = self.get_subtype_list(subtype_dict)
        subtype_mut_dict = self.get_subtype_mut_dict(subtype_list)
        return subtype_mut_dict

    def get_subtype_mut_dict(self, subtype_list):
        """
        Get subtype-mutation_dataframe dictionary

        param subtype_list:

        return:
        """
        subtype_mut_dict = dict()
        for subtype in subtype_list:
            mutation_data = self.get_mutation_feature_subtype(subtype)
            mutation_data = mutation_data.drop(columns=['gene length'])
            subtype_mut_dict[subtype] = mutation_data
        return subtype_mut_dict

    def get_mutation_feature_subtype(self, subtype):
        """
        Load subtype mutation data in DataFrame

        param subtype:

        return mutation DataFrame
        """
        mutation_data = pd.read_csv(self.mutation_path + '{}_mutation_feature.txt'.format(subtype),
                                    sep='\t', index_col=0)
        return mutation_data

    @staticmethod
    def get_subtype_list(subtype_dict):
        """
        Get subtype list including 'Tumor', union of subtypes

        param subtype_dict:

        return 'list': subtype list
        """
        subtype_list = [subtype for subtype in subtype_dict.keys()]
        subtype_list.append('Tumor')
        return subtype_list

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
