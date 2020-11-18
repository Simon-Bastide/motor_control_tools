import scipy as sp
from statsmodels.stats.anova import AnovaRM
import itertools
import pandas as pd
import numpy as np

def AnovaRM_with_post_hoc(data, dep_var, subject, within, only_significant = False):
    # One within
    anova = AnovaRM(data, dep_var, subject, within)
    print(anova.fit())
    # Post-hoc with ttest
    pairwise_ttest_rel(data,
                       dep_var,
                       within = within,
                       only_significant = only_significant
                      )        


def pairwise_ttest_rel(data, dep_var, within, only_significant = False, only_first_within_comprisons = True):
    # ttest related measures - One indep_var
    if len(within) == 1:
        conditions = data[within[0]].unique()
        list_of_ttests = list(itertools.combinations(conditions, 2))
    elif len(within) == 2:
        list1 = data[within[0]].unique()
        list2 = data[within[1]].unique()
        list_product = list(itertools.product(list1,list2))
        list_of_ttests = list(itertools.combinations(list_product, 2))
        
    print("             Post Hoc inter {}\n==========================================================================".format(' and '.join(within)))
    print("{:<48}{:>12} {:>12}".format('Test', 'p-value', 't-value'))
    indep_var = within[0]
    for combination_of_conditions in list_of_ttests:
        if len(within) == 1:
            query1 = indep_var + "==" + "'" + combination_of_conditions[0] + "'"
            query2 = indep_var + "==" + "'" + combination_of_conditions[1] + "'"
            at_least_one_same_cond = True
        elif len(within) == 2:
            at_least_one_same_cond = (combination_of_conditions[0][0] == combination_of_conditions[1][0]) or (combination_of_conditions[0][1] == combination_of_conditions[1][1])
            other_indep_var = within[1]
            query1 = indep_var + "==" "'" + combination_of_conditions[0][0] + "' & " + other_indep_var + "==" "'" + combination_of_conditions[0][1] + "'"
            query2 = indep_var + "==" "'" + combination_of_conditions[1][0] + "' & " + other_indep_var + "==" "'" + combination_of_conditions[1][1] + "'"
        
        if at_least_one_same_cond and only_first_within_comprisons:
            ttest = sp.stats.ttest_rel(data.query(query1)[dep_var],
                                        data.query(query2)[dep_var])
            
            if len(within) == 1:
                sep = ''
            elif len(within) == 2:
                sep = ' '

            if ttest.pvalue <= 0.05:
                print("\033[91m{:>22} VS {:<22}{:>12.3f}{:>12.3f}\033[0m".format(sep.join(combination_of_conditions[0]), sep.join(combination_of_conditions[1]),
                                                                   ttest.pvalue,ttest.statistic))
            elif not only_significant:
                print("{:>22} VS {:<22}{:>12.3f}{:>12.3f}".format(sep.join(combination_of_conditions[0]), sep.join(combination_of_conditions[1]),
                                                                   ttest.pvalue,ttest.statistic))


    print("==========================================================================\n\n")

    

def remove_outliers(df, columns = ['all'], zscore = 3):
    new_df = pd.DataFrame()
    for colName, colData in df.iteritems():
        if columns == ['all']:
            outliers = np.abs(sp.stats.zscore(colData)) < zscore
            data_without_outliers = colData[outliers]
            new_df = new_df.assign(**{colName : data_without_outliers})
        else:
            if colName in columns:
                outliers = np.abs(sp.stats.zscore(colData)) < zscore
                data_without_outliers = colData[outliers]
                new_df = new_df.assign(**{colName : data_without_outliers})
            else:
                new_df = new_df.assign(**{colName : colData})
    return new_df