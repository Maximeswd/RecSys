"""
Codes for running the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import argparse
import warnings
import os
import yaml
import optuna

import tensorflow as tf

from collections import defaultdict

from tuner import Tuner

parser = argparse.ArgumentParser()
possible_model_names = ["bpr", "ubpr", "wmf", "expomf", "relmf", "dubpr", "dumf"]
parser.add_argument("--models", "-m", nargs= "+", type=str, required=True, choices=possible_model_names)
parser.add_argument("--run_sims", "-r", type=int, default=10, required=True)
parser.add_argument("--datasets", "-d", nargs='+', type=str, required=True, choices=["coat", "yahoo"])
parser.add_argument('--propensity', '-p', nargs='+', type=str, required=True, choices=['original', 'bb-item', 'bb-item-user'])


def tune(trial, model_name, data, batch_size, max_iters, eta, run_sims, propensity):
    dim = trial.suggest_int("dim", 100, 300, step=20)
    lam = trial.suggest_float("lam", 1e-7, 1e-3, log=True)
    hyper_params = {
        "dim": dim,
        "lam": lam,
    }
    if model_name == "wmf":
         weight = trial.suggest_float("weight", 1e-3, 1)
         hyper_params['weight'] = weight
    elif model_name in ["ubpr", 'dubpr']:
        beta = trial.suggest_float("beta", 1e-2, 10)
        hyper_params['beta'] = beta
    elif model_name in ["relmf", 'dumf']:     
        clip = trial.suggest_float("clip", 0, 0.5)
        hyper_params['clip'] = clip
       
    tuner = Tuner(
        data=data,
        batch_size=batch_size,
        max_iters=max_iters,
        eta=eta,
        model_name=model_name,
        hyper_params=hyper_params,
        propensity=propensity
    )
    result = tuner.run(num_sims=run_sims)
    return result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()
    config = yaml.safe_load(open("../conf/config.yaml", "rb"))
    for propensity in args.propensity:
        hyper_params_file = f"../conf/{propensity}_hyper_params.yaml"
        if os.path.isfile(hyper_params_file):
            tuned_params = defaultdict(dict, yaml.safe_load(open(hyper_params_file, "rb")))
        else:
             tuned_params = defaultdict(dict)
        for data in args.datasets:
            model_tuned_params= tuned_params[data]
            for model in args.models:
                    objective = lambda trial: tune(
                        trial,
                        model,
                        data,
                        config["batch_size"],
                        config["max_iters"],
                        config["eta"],
                        args.run_sims,
                        propensity
                    )
                    study = optuna.create_study(direction="minimize")
                    study.optimize(objective, n_trials=100)
                    model_tuned_params[model] = study.best_params             
            tuned_params[data] = model_tuned_params
        with open(hyper_params_file, 'w') as file:
            yaml.dump(data=dict(tuned_params), stream=file)