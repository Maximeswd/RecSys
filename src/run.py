import argparse
import subprocess

parser = argparse.ArgumentParser(description='Run models on different datasets with different loss functions.')
parser.add_argument('-d', '--datasets', nargs='+', help='Datasets to use', required=True)
parser.add_argument('-m', '--models', nargs='+', help='Models to use', required=True)
parser.add_argument('--pointwise_loss', type=str, help='Pointwise loss function to use', required=True, choices=['original', 'cross_entropy','dual_unbiased'])
parser.add_argument('--pairwise_loss', type=str, help='Pairwise loss function to use', required=True, choices=['original', 'dual_unbiased'])
parser.add_argument('-p', '--propensity', nargs='+', help='Propensity to use', type=str, required=True, choices=['original', 'bb-item', 'bb-item-user'])
parser.add_argument('-r', '--run_sims', help='Number of simulations', type=int, required=True)
args = parser.parse_args()

for data in args.datasets:
    for model in args.models:
        for propensity in args.propensity:
            cmd = f"python main.py -m {model} -d {data} -r {args.run_sims} --pointwise_loss {args.pointwise_loss} --pairwise_loss {args.pairwise_loss} -p {propensity}"
            subprocess.run(cmd, shell=True, check=True)