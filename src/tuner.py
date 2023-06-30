"""
# Implemented by Ilse/Maxime/Abhijith
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
    def __init__(
        self,
        data: str,
        max_iters: int = 1000,
        batch_size: int = 12,
        eta: float = 0.1,
        model_name: str = "bpr",
        hyper_params: dict = None,
        propensity: str = 'original'
    ) -> None:
        """Initialize class."""
        self.data = data
        self.propensity = propensity  

        self.dim = np.int(hyper_params["dim"])
        self.lam = hyper_params["lam"]
        self.weight = hyper_params["weight"] if model_name == "wmf" else 1.0
        self.clip = hyper_params["clip"] if model_name in ["relmf", 'dumf'] else 0.0
        self.beta = hyper_params["beta"] if model_name in ["ubpr", 'dubpr'] else 0.0

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eta = eta
        self.model_name = model_name

    def run(self, num_sims: int = 10) -> None:
        """Tuning the models based on validation loss."""
        train_point = np.load(f"../data/{self.data}/{self.propensity}/point/train.npy")
        val_point = np.load(f"../data/{self.data}/{self.propensity}/point/val.npy")
        test_point = np.load(f"../data/{self.data}/{self.propensity}/point/test.npy")
        pscore = np.load(f"../data/{self.data}/{self.propensity}/point/pscore.npy")
        nscore = np.load(f"../data/{self.data}/{self.propensity}/point/nscore.npy")
        num_users = np.int(train_point[:, 0].max() + 1)
        num_items = np.int(train_point[:, 1].max() + 1)
        if self.model_name in ["bpr", "ubpr", "dubpr"]:
            train = np.load(f"../data/{self.data}/{self.propensity}/pair/{self.model_name}_train.npy")
            val = np.load(f"../data/{self.data}/{self.propensity}/pair/{self.model_name}_val.npy")
            test = np.load(f"../data/{self.data}/{self.propensity}/pair/test.npy")
        if self.data == "yahoo":
            user_freq = np.load(f"../data/{self.data}/{self.propensity}/point/user_freq.npy")
            item_freq = np.load(f"../data/{self.data}/{self.propensity}/point/item_freq.npy")
        
        val_losses = []
        for seed in np.arange(num_sims):
            ops.reset_default_graph()

            tf.set_random_seed(seed)
            np.random.seed(seed)

            sess = tf.Session()
            if self.model_name in ["ubpr", "bpr", "dubpr"]:
                if self.model_name in ["ubpr", "bpr"]:
                    pair_rec = PairwiseRecommender(
                        num_users=num_users,
                        num_items=num_items,
                        dim=self.dim,
                        lam=self.lam,
                        eta=self.eta,
                        beta=self.beta,
                    )
                elif self.model_name == 'dubpr':
                    pair_rec = PairwiseRecommender(
                        num_users=num_users,
                        num_items=num_items,
                        dim=self.dim,
                        lam=self.lam,
                        eta=self.eta,
                        beta=self.beta,
                        loss_function='dual_unbiased'
                    )

                u_emb, i_emb, val_loss = train_pairwise(
                    sess,
                    model=pair_rec,
                    data=self.data,
                    train=train,
                    val=val,
                    test=test,
                    max_iters=self.max_iters,
                    batch_size=self.batch_size,
                    model_name=self.model_name,
                    propensity=self.propensity,
                    seed=seed
                )
            elif self.model_name in ["wmf", "relmf", 'dumf']:
                if self.model_name in ["wmf", "relmf"]:
                    point_rec = PointwiseRecommender(
                        num_users=num_users,
                        num_items=num_items,
                        weight=self.weight,
                        clip=self.clip,
                        dim=self.dim,
                        lam=self.lam,
                        eta=self.eta,
                    )
                elif self.model_name == 'dumf':
                    point_rec = PointwiseRecommender(
                        num_users=num_users,
                        num_items=num_items,
                        weight=self.weight,
                        clip=self.clip,
                        dim=self.dim,
                        lam=self.lam,
                        eta=self.eta,
                        loss_function='dual_unbiased'
                    )
                u_emb, i_emb, val_loss = train_pointwise(
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
                        propensity=self.propensity,
                        nscore=nscore,
                        is_optuna=True
                    )
            val_losses.append(val_loss)        
        
        return sum(val_losses)/len(val_losses)
