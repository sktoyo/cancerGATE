from PreprocessExpression import *
from PreprocessMutation import *
from PreprocessNetwork import *

def preprocess():
    """
    Save feature matrix and network pickle
    """
    preExp = PreprocessExpression('breast', 'brca')
    subtype_exp_dict = preExp.preprocess()

    print(subtype_exp_dict['Normal'].head())
    print(subtype_exp_dict['Tumor'].head())




preprocess()
