"""
This script is made to run the statistical comparison of methods and propensity estimation of the paper: A Replication Study: 
Unbiased Pairwise Learning from Biased Implicit Feedback
Made by Ilse Feenstra, Maxime Dassen and Abijith Chintam
"""
import argparse
import warnings
import os
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from itertools import combinations

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', '-d', nargs='+', type=str, required=True, choices=['coat', 'yahoo'])
parser.add_argument('--propensity', '-p', nargs='+', type=str, required=True, choices=['original', 'bb-item-user'])
parser.add_argument('--methods', '-m', nargs='+', type=str, required=True, choices=['wmf', 'relmf', 'bpr', 'ubpr', 'dumf', 'dubpr']) # ip and expomf are not included because they don't work with the MANOVA test since all iterations are the same
parser.add_argument('--comparison', '-c', nargs=1, help='parse one comparison at the time', type=str, required=True, choices=['method', 'prop'])

def load_data(logs_folder, methods):
    """ This function loads the right data """
    
    dfs = []
    propensities = ['original', 'bb-item-user']

    if not isinstance(propensities, list):
        propensities == ['original']
        print(propensities[0])

    for propensity in propensities:
        for method in methods:
            file_path = os.path.join(logs_folder, propensity, method, "results/aoa_all.csv")
            print(file_path)

            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
                df['method'] = method
                df['prop'] = propensity
                dfs.append(df)
    
    return dfs


def prepare_data(df, methods):
    """ This function turns the data into the right format for the MANOVA analysis """
    df_prepared = pd.DataFrame()
    df.rename(columns={'Unnamed: 0': 'metric'}, inplace=True)
    df['metric'] = df['metric'].str.replace('@', '_')

    for method in methods:
        # Define the columns associated with the method
        cols = [method] + [method + '.' + str(i) for i in range(1, 10)]

        # Select only the relevant columns
        df_method = df[['metric', 'prop'] + cols].copy()

        # Rename the columns
        df_method.columns = [df_method.columns[0]] + [df_method.columns[1]] + ['iteration_' + str(i) for i in range(len(df_method.columns)-2)]

        # Add 'method' column
        df_method['method'] = method

        # Melt the DataFrame to long format
        df_method = df_method.melt(id_vars=['metric', 'method', 'prop'], var_name='iteration', value_name='value')
        df_method = df_method.dropna()
    
        df_prepared = pd.concat([df_prepared, df_method], ignore_index=True)

    return df_prepared


def manova_test_method(df):
    """ This function performs the MANOVA test between methods per dataset per propensity estimation """
    
    df_wide = df.pivot_table(index=['method', 'iteration'], columns='metric', values='value').reset_index()
    manova = MANOVA.from_formula('DCG_3 + Recall_3 + MAP_3 + DCG_5 + Recall_5 + MAP_5 + DCG_8 + Recall_8 + MAP_8 ~ method', data=df_wide)
    print(manova.mv_test())

def manova_test_prop(df):
    """ This function performs the the MANOVA test between propensity estimations per dataset per method """
    
    methods = ['bpr', 'ubpr']

    for method in methods: 
        print(f'MANOVA for method {method}')
        df_method = df[df['method'] == method]
        df_method = df_method.drop(columns =['method'], axis=1)
        df_wide = df_method.pivot_table(index=['prop', 'iteration'], columns='metric', values='value').reset_index()

        manova = MANOVA.from_formula('DCG_3 + Recall_3 + MAP_3 + DCG_5 + Recall_5 + MAP_5 + DCG_8 + Recall_8 + MAP_8 ~ prop', data=df_wide)
        print(manova.mv_test())

def post_hoc_test_methods(df_prepared):
    """ This function performs the Tukey HSD test and does a pairwise comparison between methods """
    
    methods = df_prepared['method'].unique()
    metrics = df_prepared['metric'].unique()

    results = []

    for metric in metrics:

        df_metric = df_prepared[df_prepared['metric'] == metric]

        for method1, method2 in combinations(methods, 2):
            df_pair = df_metric[(df_metric['method'] == method1) | (df_metric['method'] == method2)]
            model = ols('value ~ C(method)', data=df_pair).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_value = anova_table['PR(>F)']['C(method)']
            results.append([metric, method1, method2, p_value])

    df_results = pd.DataFrame(results, columns=['metric', 'method1', 'method2', 'p_value'])
    df_results['significant'] = df_results['p_value'] < 0.05  
    print(df_results)

    
    for metric in metrics:

        print(f"Tukey's HSD for {metric}")
        df_metric = df_prepared[df_prepared['metric'] == metric]
        posthoc = pairwise_tukeyhsd(df_metric['value'], df_metric['method'], alpha=0.05)
        print(posthoc)

def post_hoc_test_prop(df_prepared):
    """ This function performs the Tukey HSD test and does a pairwise comparison between propensity estimations """

    methods = df_prepared['method'].unique()
    metrics = df_prepared['metric'].unique()
    results = []

    for method in methods:
        for metric in metrics:
    
            df_method_metric = df_prepared[(df_prepared['method'] == method) & (df_prepared['metric'] == metric)]

            # Propensity estimations to compare
            prop1 = 'original'
            prop2 = 'bb-item-user'
            
            df_pair = df_method_metric[(df_method_metric['prop'] == prop1) | (df_method_metric['prop'] == prop2)]
            model = ols('value ~ C(prop)', data=df_pair).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_value = anova_table['PR(>F)']['C(prop)']
            results.append([method, metric, prop1, prop2, p_value])

    df_results = pd.DataFrame(results, columns=['method', 'metric', 'prop1', 'prop2', 'p_value'])
    df_results['significant'] = df_results['p_value'] < 0.05  # or apply the Bonferroni correction here
    print(df_results)

    for method in methods:

        for metric in metrics:
            df_method_metric = df_prepared[(df_prepared['method'] == method) & (df_prepared['metric'] == metric)]
            posthoc = pairwise_tukeyhsd(df_method_metric['value'], df_method_metric['prop'], alpha=0.05)
            print(f"Tukey's HSD for {method} - {metric}:")
            print(posthoc)

def main(dataset, methods, comparison):
    
    logs_folder = f'../logs/{dataset}/'
    dfs = load_data(logs_folder, methods)
    df_concat = pd.concat(dfs)

    
    df_prepared = prepare_data(df_concat, methods)

    if 'method' in comparison:
        manova_test_method(df_prepared)
        post_hoc_test_methods(df_prepared)
    
    elif 'prop' in comparison:

        manova_test_prop(df_prepared)
        post_hoc_test_prop(df_prepared)

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    comparison = args.comparison
    
    for data in args.datasets:

        if 'method' in comparison:
            for propensity in args.propensity:
                main(data, args.methods, comparison)

        elif 'prop' in comparison:
            main(data, args.methods, comparison)