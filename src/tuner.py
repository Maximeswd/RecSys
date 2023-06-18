"""
Codes for training recommenders used in the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import yaml
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import sparse
from tensorflow.python.framework import ops

from evaluate.evaluator import aoa_evaluator
from models.expomf import ExpoMF
from models.recommenders import PairwiseRecommender, PointwiseRecommender

from trainer import train_expomf, train_pairwise, train_pointwise


class Tuner:
    suffixes = ["cold-user", "rare-item"]
    at_k = [3, 5, 8]
    cold_user_threshold = 6
    rare_item_threshold = 100

    def __init__(
        self,
        data: str,
        max_iters: int = 1000,
        batch_size: int = 12,
        eta: float = 0.1,
        model_name: str = "bpr",
        hyper_params: dict = None,
    ) -> None:
        """Initialize class."""
        self.data = data
        if model_name != "expomf":
            self.dim = np.int(hyper_params["dim"])
            self.lam = hyper_params["lam"]
            self.weight = hyper_params["weight"] if model_name == "wmf" else 1.0
            self.clip = hyper_params["clip"] if model_name == "relmf" else 0.0
            self.beta = hyper_params["beta"] if model_name == "ubpr" else 0.0
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eta = eta
        self.model_name = model_name

    def run(self, num_sims: int = 10, metric="DCG@3") -> None:
        """Train implicit recommenders."""
        train_point = np.load(f"../data/{self.data}/point/train.npy")
        val_point = np.load(f"../data/{self.data}/point/val.npy")
        test_point = np.load(f"../data/{self.data}/point/test.npy")
        pscore = np.load(f"../data/{self.data}/point/pscore.npy")
        num_users = np.int(train_point[:, 0].max() + 1)
        num_items = np.int(train_point[:, 1].max() + 1)
        if self.model_name in ["bpr", "ubpr", "dubpr"]:
            train = np.load(f"../data/{self.data}/pair/{self.model_name}_train.npy")
            val = np.load(f"../data/{self.data}/pair/{self.model_name}_val.npy")
            test = np.load(f"../data/{self.data}/pair/test.npy")
        if self.data == "yahoo":
            user_freq = np.load(f"../data/{self.data}/point/user_freq.npy")
            item_freq = np.load(f"../data/{self.data}/point/item_freq.npy")

        result_list = list()
        if self.data == "yahoo":
            cold_user_result_list = list()
            rare_item_result_list = list()
        for seed in np.arange(num_sims):
            tf.set_random_seed(12345)
            ops.reset_default_graph()
            sess = tf.Session()
            if self.model_name in ["ubpr", "bpr", "dubpr"]:
                pair_rec = PairwiseRecommender(
                    num_users=num_users,
                    num_items=num_items,
                    dim=self.dim,
                    lam=self.lam,
                    eta=self.eta,
                    beta=self.beta,
                )
                u_emb, i_emb, _ = train_pairwise(
                    sess,
                    model=pair_rec,
                    data=self.data,
                    train=train,
                    val=val,
                    test=test,
                    max_iters=self.max_iters,
                    batch_size=self.batch_size,
                    model_name=self.model_name,
                )
            elif self.model_name in ["wmf", "relmf"]:
                point_rec = PointwiseRecommender(
                    num_users=num_users,
                    num_items=num_items,
                    weight=self.weight,
                    clip=self.clip,
                    dim=self.dim,
                    lam=self.lam,
                    eta=self.eta,
                )
                u_emb, i_emb, _ = train_pointwise(
                    sess,
                    model=point_rec,
                    data=self.data,
                    train=train_point,
                    val=val_point,
                    test=test_point,
                    pscore=pscore,
                    max_iters=self.max_iters,
                    batch_size=self.batch_size,
                    model_name=self.model_name,
                )
            elif self.model_name == "expomf":
                u_emb, i_emb = train_expomf(
                    data=self.data,
                    train=train_point,
                    num_users=num_users,
                    num_items=num_items,
                )

            result = aoa_evaluator(
                user_embed=u_emb,
                item_embed=i_emb,
                test=test_point,
                model_name=self.model_name,
                at_k=self.at_k,
            )
            result_list.append(result)

            if self.data == "yahoo":
                user_idx, item_idx = test_point[:, 0].astype(int), test_point[
                    :, 1
                ].astype(int)
                cold_user_idx = user_freq[user_idx] <= self.cold_user_threshold
                rare_item_idx = item_freq[item_idx] <= self.rare_item_threshold
                cold_user_result = aoa_evaluator(
                    user_embed=u_emb,
                    item_embed=i_emb,
                    at_k=self.at_k,
                    test=test_point[cold_user_idx],
                    model_name=self.model_name,
                )
                rare_item_result = aoa_evaluator(
                    user_embed=u_emb,
                    item_embed=i_emb,
                    at_k=self.at_k,
                    test=test_point[rare_item_idx],
                    model_name=self.model_name,
                )
                cold_user_result_list.append(cold_user_result)
                rare_item_result_list.append(rare_item_result)

            print(f"#{seed+1}: {self.model_name}...")

        ret_path = Path(f"../logs/{self.data}/{self.model_name}/results")
        ret_path.mkdir(parents=True, exist_ok=True)
        pd.concat(result_list, 1).to_csv(ret_path / f"aoa_all.csv")
        if self.data == "yahoo":
            pd.concat(cold_user_result_list, 1).to_csv(ret_path / f"aoa_cold-user.csv")
            pd.concat(rare_item_result_list, 1).to_csv(ret_path / f"aoa_rare-item.csv")

        results_all = pd.concat(result_list, 1)
        results_all["mean"] = results_all.mean(axis=1)
        print(results_all.head())
        print("Average DCG@5: ", results_all.loc["DCG@5", "mean"])
        return results_all.loc["DCG@5", "mean"]
