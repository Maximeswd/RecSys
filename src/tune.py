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
possible_model_names = ["bpr", "ubpr", "wmf", "expomf", "relmf", "dubpr"]
parser.add_argument(
    "--model_name", "-m", type=str, required=True, choices=possible_model_names
)
parser.add_argument("--run_sims", "-r", type=int, default=10, required=True)
parser.add_argument("--data", "-d", type=str, required=True, choices=["coat", "yahoo"])


def tune(trial, model_name, data, metric, batch_size, max_iters, eta, run_sims):
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
    )
    result = tuner.run(num_sims=run_sims, metric=metric)
    return result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()
    config = yaml.safe_load(open("../conf/config.yaml", "rb"))

    objective = lambda trial: tune(
        trial,
        args.model_name,
        args.data,
        "DCG@3",
        config["batch_size"],
        config["max_iters"],
        config["eta"],
        args.run_sims,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print(study.best_params)

    print("\n", "=" * 25, "\n")
    print(f"Finished tuning {args.model_name}!")
    print("\n", "=" * 25, "\n")
