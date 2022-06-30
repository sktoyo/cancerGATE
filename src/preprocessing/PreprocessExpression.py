import pandas as pd
import os

class PreprocessExpression():
    def __init__(self, expression_path, tissue):
        self.expression_path = expression_path
        self.tissue = tissue

    def preprocess(self):
        """
        Preprocess the TCGA TPM data and GTEx TPM data to average data

        return 'dict': expression dict with dataframe format
        """
        cancer_df = self.get_tpm_df(self.expression_path + "TCGA_BRCA_TPM_symbol.tsv")
        gtex_exp_df = self.get_gtex_exp_df()


    gtex_exp_df = gtex_exp_df[gtex_exp_df.astype('bool').mean(axis=1) >= 0.2]
    gtex_exp_df = gtex_exp_df.groupby('Description').mean()

    # intersection of genes between TCGA and GTEx
    expression_df = get_tpm_df(expression_path)

    intersection_gene = expression_df.index.intersection(gtex_exp_df.index)
    expression_df = expression_df.loc[intersection_gene]
    expression_df = expression_df.sort_index()

    gtex_exp_df = gtex_exp_df.loc[intersection_gene]
    gtex_exp_df = gtex_exp_df.sort_index()

    # filter tumor samples shared with mutation data
    sample_list = get_sample_list()
    expression_df = expression_df.loc[:, expression_df.columns.isin(sample_list)]

    # create subtype expression DF dictionary
    orig_subtype_dict = getSubtypeDict()
    subtype_dict = dict()
    for key, value in orig_subtype_dict.items():
        if value in subtype_dict.keys():
            subtype_dict[value].append(key)
        else:
            subtype_dict[value] = [key]

    subtype_expression_dict = dict()
    for subtype in subtype_dict.keys():
        subtype_expression = isolate_sample(expression_df, subtype_dict[subtype])
        subtype_expression_dict[subtype] = subtype_expression

    subtype_expression_dict['Normal'] = gtex_exp_df  # Change TCGA normal -> GTEx normal

    tumor_expression = None
    for subtype in subtype_expression_dict.keys():
        if subtype == 'Normal':
            continue
        elif tumor_expression is None:
            tumor_expression = subtype_expression_dict[subtype]
        else:
            tumor_expression = pd.concat([tumor_expression, subtype_expression_dict[subtype]], axis=1)

    subtype_expression_dict['Tumor'] = tumor_expression

    average = True
    if average:
        for subtype in subtype_expression_dict:
            avg_series = subtype_expression_dict[subtype].mean(axis=1)
            subtype_expression_dict[subtype] = pd.DataFrame(avg_series, columns=['avg_TPM'])

    def get_gtex_exp_df(self):
        """
        Get the GTEx TPM data, filter related tissue only dataframe

        return 'dict': expression dict with dataframe format
        """
        if self.check_gtex_file():
            gtex_exp_df = pd.read_csv(self.expression_path + "gtex_{}_exp.tsv".format(self.tissue), sep='\t', index_col=0)
        else:
            gtex_exp_df = self.get_tissue_gtex(self)
        return gtex_exp_df

    def check_gtex_file(self):
        """
        Check whether file exists
        return: 'bool'
        """
        if os.path.exists(self.expression_path + "gtex_{}_exp.tsv".format(self.tissue)):
            return True
        else:
            return False

    def get_tissue_gtex(self):
        """
        filter gtex expression with tissue
        return:
        """
        gtex_df = self.get_tpm_df_gtex(self.expression_path + "GTEx_tpm.gct")
        gtex_label_info = self.get_gtex_label_info()

        gtex_sample_list = gtex_label_info[gtex_label_info['SMTS'] == self.tissue].index.to_list()
        gtex_sample_list_filtered = [sample for sample in gtex_sample_list if sample in gtex_df.columns.tolist()]
        gtex_sample_list_filtered = ['Description'] + gtex_sample_list_filtered
        gtex_exp_df = gtex_df[gtex_sample_list_filtered]
        gtex_exp_df.to_csv(self.expression_path + "gtex_{}_exp.tsv".format(self.tissue), sep="\t")
        return gtex_exp_df

    def get_gtex_label_info(self):
        """
        Get gtex tissue label information

        return: gtex label information
        """
        gtex_label_info = pd.read_csv(self.expression_path + 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt',
                                      sep='\t',index_col=0)
        return gtex_label_info

    def get_tpm_df_gtex(self):
        """
        Get all gtex tpm datafrme

        return: gtex tpm dataframe
        """
        expression_df = pd.read_csv(self.gtex_path, sep='\t', index_col=0, skiprows=2)
        return expression_df

    def get_tpm_df(self):
        """
        Get all gtex tpm datafrme

        return: gtex tpm dataframe
        """
        expression_df = pd.read_csv(self.expression_path + 'TCGA_BRCA_TPM_symbol.tsv', sep='\t', index_col=0)
        return expression_df

    def get_subtype_dict(self):
        """
        Get subtype information of TCGA BRCA

        return: subtype dictionary
        """
        file_info_df = pd.read_csv(self.expression_path + "TCGA_BRCA_mol_subtype.tsv", sep='\t')
        subtype_dict = dict()

        for index, row in file_info_df.iterrows():
            subtype = row['Subtype_mRNA']
            sample_id = row['pan.samplesID']
            sample_id = '-'.join(sample_id.split('-')[:4])
            subtype_dict[sample_id] = subtype

        return subtype_dict
