"""
Codes for running the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import argparse
import warnings
import yaml
import optuna

import tensorflow as tf

from tuner import Tuner

parser = argparse.ArgumentParser()
possible_model_names = ["bpr", "ubpr", "wmf", "expomf", "relmf", "dubpr", "dumf"]
parser.add_argument("--models", "-m", nargs= "+", type=str, required=True, choices=possible_model_names)
parser.add_argument("--run_sims", "-r", type=int, default=10, required=True)
parser.add_argument("--datasets", "-d", nargs='+', type=str, required=True, choices=["coat", "yahoo"])
parser.add_argument('--propensity', '-p', nargs='+', type=str, required=True, choices=['original', 'bb-item', 'bb-item-user'])


def tune(trial, model_name, data, metric, batch_size, max_iters, eta, run_sims, propensity):
    beta = trial.suggest_float("beta", 1e-2, 10, log=True)
    dim = trial.suggest_int("dim", 100, 300, step=20)
    lam = trial.suggest_float("lam", 1e-7, 1e-3, log=True)
    hyper_params = {
        "beta": beta,
        "dim": dim,
        "lam": lam,
    }
    tuner = Tuner(
        data=data,
        batch_size=batch_size,
        max_iters=max_iters,
        eta=eta,
        model_name=model_name,
        hyper_params=hyper_params,
        propensity=propensity
    )
    result = tuner.run(num_sims=run_sims, metric=metric)
    return result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()
    config = yaml.safe_load(open("../conf/config.yaml", "rb"))

    

    for data in args.datasets:
        for model in args.models:
            for propensity in args.propensity:
                objective = lambda trial: tune(
                    trial,
                    model,
                    data,
                    "DCG@3",
                    config["batch_size"],
                    config["max_iters"],
                    config["eta"],
                    args.run_sims,
                    propensity
                )

                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=100)

                print(study.best_params)

                print("\n", "=" * 25, "\n")
                print(f"Finished tuning {args.model_name} with propensity estimation {args.propensity}!")
                print("\n", "=" * 25, "\n")


    # # debug
    # datasets = ['coat']
    # models = ['dumf']
    # propensity = ['original']
    # config = yaml.safe_load(open("../RecSys/conf/config.yaml", "rb"))
    

    # for data in datasets:
    #     for model in models:
    #         for propensity in propensity:
    #             objective = lambda trial: tune(
    #                 trial,
    #                 model,
    #                 data,
    #                 "DCG@3",
    #                 config["batch_size"],
    #                 config["max_iters"],
    #                 config["eta"],
    #                 1,
    #                 propensity
    #             )

    #             study = optuna.create_study(direction="maximize")
    #             study.optimize(objective, n_trials=100)

    #             print(study.best_params)

    #             print("\n", "=" * 25, "\n")
    #             print(f"Finished tuning {args.model_name} with propensity estimation {args.propensity}!")
    #             print("\n", "=" * 25, "\n")