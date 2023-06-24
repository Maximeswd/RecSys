"""
Codes for preprocessing the real-world datasets
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
Extended with the code from the paper "Reproducibility study: Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import argparse
import warnings

from preprocess.preprocessor import preprocess_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', '-d', nargs='+', type=str, required=True, choices=['coat', 'yahoo'])
parser.add_argument('--propensity', '-p', nargs='+', type=str, required=True, choices=['original', 'bb-item', 'bb-item-user'])

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    for data in args.datasets:
        for propensity in args.propensity:
            preprocess_dataset(data=data, propensity=propensity)

            print('\n', '=' * 25, '\n')
            print(f'Finished Preprocessing {data} for propensity method {propensity}!')
            print('\n', '=' * 25, '\n')
