"""
Codes for summarizing results of the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd


# configurations.
#all_models = ['ip', 'wmf', 'expomf', 'relmf', 'bpr', 'ubpr', 'dumf', 'dubpr', 'ngcf_ubpr', 'ngcf_bpr']
K = [3, 5, 8]
metrics = ['DCG', 'Recall', 'MAP']
col_names = [f'{m}@{k}' for m in metrics for k in K]
rel_col_names = [f'{m}@5' for m in metrics]

# Modified by Ilse/Maxime/Abhijith
def summarize_results(data: str, path: Path, propensity, all_models) -> None:
    """Load and save experimental results."""
    suffixes = ['all'] if data == 'coat' else ['cold-user', 'rare-item', 'all']
    for suffix in suffixes:
        aoa_list = []
        for model in all_models:
            file = f'../logs/{data}/{propensity}/{model}/results/aoa_{suffix}.csv'
            aoa_list.append(pd.read_csv(file, index_col=0).mean(1))
        results_df = pd.concat(aoa_list, 1).round(7).T
        results_df.index = all_models
        results_df[col_names].to_csv(path / f'ranking_{suffix}.csv')


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', '-d', required=True, nargs='*', type=str, choices=['coat', 'yahoo'])
parser.add_argument('--propensity', '-p', nargs='+', type=str, required=True, choices=['original', 'bb-item', 'bb-item-user'])
parser.add_argument('--models', '-m', required=True, nargs='*', type=str, choices=['ip', 'wmf', 'expomf', 'relmf', 'bpr', 'ubpr', 'dumf', 'dubpr', 'ngcf_ubpr', 'ngcf_bpr'])

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    for data in args.datasets:
        for propensity in args.propensity:
            path = Path(f'../paper_results/{data}/{propensity}')
            path.mkdir(parents=True, exist_ok=True)
            summarize_results(data=data, path=path, propensity=propensity, all_models=args.models)
