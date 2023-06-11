import argparse
import subprocess

parser = argparse.ArgumentParser(description='Run models on different datasets with different loss functions.')
parser.add_argument('-d', '--datasets', nargs='+', help='Datasets to use', required=True)
parser.add_argument('-m', '--models', nargs='+', help='Models to use', required=True)
parser.add_argument('--pointwise_loss', type=str, help='Pointwise loss function to use', required=True)
parser.add_argument('--pairwise_loss', type=str, help='Pairwise loss function to use', required=True)
args = parser.parse_args()

for data in args.datasets:
    for model in args.models:
        cmd = f"python main.py -m {model} -d {data} -r 10 --pointwise_loss {args.pointwise_loss} --pairwise_loss {args.pairwise_loss}"
        subprocess.run(cmd, shell=True, check=True)