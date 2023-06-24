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


def train_ip(num_users: int, num_items: int, scores: np.ndarray,) -> Tuple:
    return np.ones(shape=(num_users, 1)), scores.reshape(num_items, 1)

def train_expomf(
    data: str,
    train: np.ndarray,
    num_users: int,
    num_items: int,
    n_components: int = 100,
    lam: float = 1e-6,
    
) -> Tuple:
    """Train the expomf model."""

    def tocsr(data: np.array, num_user: int, num_item: int) -> sparse.csr_matrix:
        """Convert data to csr_matrix."""
        matrix = sparse.lil_matrix((num_users, num_items))
        for (u, i, r) in data[:, :3]:
            matrix[u, i] = r
        return sparse.csr_matrix(matrix)

    path = Path(f"../logs/{data}/expomf/emb")
    path.mkdir(parents=True, exist_ok=True)
    model = ExpoMF(
        n_components=n_components,
        random_state=12345,
        save_params=False,
        early_stopping=True,
        verbose=False,
        lam_theta=lam,
        lam_beta=lam,
    )
    model.fit(tocsr(train, num_users, num_items))
    np.save(file=str(path / "user_embed.npy"), arr=model.theta)
    np.save(file=str(path / "item_embed.npy"), arr=model.beta)

    return model.theta, model.beta


def train_pointwise(
    sess: tf.Session,
    model: PointwiseRecommender,
    data: str,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    pscore: np.ndarray,
    nscore: np.ndarray,
    max_iters: int = 1000,
    batch_size: int = 256,
    model_name: str = "wmf",
    is_optuna: bool = False,
    loss = str
) -> Tuple:
    """Train and evaluate implicit recommender."""
    train_loss_list = []
    test_loss_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    ips = model_name == "relmf" or "dumf"
    # pscore for train
    pscore = pscore[train[:, 1].astype(int)]

    # nscore for train
    nscore = nscore[train[:, 1].astype(int)]
   
    # positive and unlabeled data for training set
    pos_train = train[train[:, 2] == 1]
    pscore_pos_train = pscore[train[:, 2] == 1]
    num_pos = np.sum(train[:, 2])
    unlabeled_train = train[train[:, 2] == 0]
    pscore_unlabeled_train = pscore[train[:, 2] == 0]

    num_unlabeled = np.sum(1 - train[:, 2])
    
    # train the given implicit recommender
    np.random.seed(12345)  # TODO random seed is not in loop
    for i in np.arange(max_iters):
        # positive mini-batch sampling
        # the same num. of postive and negative samples are used in each batch
        
        sample_size = np.int(batch_size / 2)
        pos_idx = np.random.choice(np.arange(num_pos), size=sample_size)
        unl_idx = np.random.choice(np.arange(num_unlabeled), size=sample_size)
        
        # mini-batch samples
        train_batch = np.r_[pos_train[pos_idx], unlabeled_train[unl_idx]]
        pscore_ = (
            np.r_[pscore_pos_train[pos_idx], pscore_unlabeled_train[unl_idx]]
            if ips
            else np.ones(batch_size)
        )

        nscore_ = (
            np.r_[pscore_unlabeled_train[unl_idx], pscore_pos_train[pos_idx]]
            if ips
            else np.zeros(batch_size)
        )


        # update user-item latent factors and calculate training loss
        _, train_loss = sess.run(
            [model.apply_grads, model.loss],
            feed_dict={
                model.users: train_batch[:, 0],
                model.items: train_batch[:, 1],
                model.labels: np.expand_dims(train_batch[:, 2], 1),
                model.scores: np.expand_dims(pscore_, 1),
                model.pscores: np.expand_dims(pscore_, 1),
                model.nscores: np.expand_dims(nscore_, 1) 
            },
        )
        train_loss_list.append(train_loss)

    # calculate a validation score
    unl_idx = np.random.choice(np.arange(num_unlabeled), size=val.shape[0])
    pos_idx = np.random.choice(np.arange(num_pos), size=val.shape[0])
    val_batch = np.r_[val, unlabeled_train[unl_idx]]
    pscore_ = np.r_[pscore[val[:, 1].astype(int)], pscore_unlabeled_train[unl_idx]]
    nscore_ = np.r_[nscore[val[:, 1].astype(int)], pscore[val[:, 1].astype(int)]]
    val_loss = sess.run(
        model.loss,
        feed_dict={
            model.users: val_batch[:, 0],
            model.items: val_batch[:, 1],
            model.labels: np.expand_dims(val_batch[:, 2], 1),
            model.scores: np.expand_dims(pscore_, 1),
            model.pscores: np.expand_dims(pscore_, 1),
            model.nscores: np.expand_dims(nscore_, 1) 
        },
    )

    u_emb, i_emb = sess.run([model.user_embeddings, model.item_embeddings])
    if ~is_optuna:
        path = Path(f"../logs/{data}/{model_name}")
        (path / "loss").mkdir(parents=True, exist_ok=True)
        np.save(file=str(path / "loss/train.npy"), arr=train_loss_list)
        np.save(file=str(path / "loss/test.npy"), arr=test_loss_list)
        (path / "emb").mkdir(parents=True, exist_ok=True)
        np.save(file=str(path / "emb/user_embed.npy"), arr=u_emb)
        np.save(file=str(path / "emb/item_embed.npy"), arr=i_emb)
    sess.close()

    return u_emb, i_emb, val_loss


def train_pairwise(
    sess: tf.Session,
    model: PairwiseRecommender,
    data: str,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    max_iters: int = 1000,
    batch_size: int = 1024,
    model_name: str = "bpr",
    is_optuna: bool = False,
) -> Tuple:
    """Train and evaluate pairwise recommenders."""
    train_loss_list = []
    test_loss_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # count the num of training data.
    num_train, num_val = train.shape[0], val.shape[0]
    np.random.seed(12345)
    for i in np.arange(max_iters):
        idx = np.random.choice(np.arange(num_train), size=batch_size)
        train_batch = train[idx]
        # update user-item latent factors
        if model_name in "bpr":
            _, loss = sess.run(
                [model.apply_grads, model.loss],
                feed_dict={
                    model.users: train_batch[:, 0],
                    model.pos_items: train_batch[:, 1],
                    model.scores1: np.ones((batch_size, 1)),
                    model.items2: train_batch[:, 2],
                    model.labels2: np.zeros((batch_size, 1)),
                    model.scores2: np.ones((batch_size, 1)),
                },
            )
        elif model_name in "ubpr":
            _, loss = sess.run(
                [model.apply_grads, model.loss],
                feed_dict={
                    model.users: train_batch[:, 0],
                    model.pos_items: train_batch[:, 1],
                    model.scores1: np.expand_dims(train_batch[:, 4], 1),
                    model.items2: train_batch[:, 2],
                    model.labels2: np.expand_dims(train_batch[:, 3], 1),
                    model.scores2: np.expand_dims(train_batch[:, 5], 1),
                },
            )
        
        elif model_name in "dubpr":
            _, loss = sess.run(
            [model.apply_grads, model.loss],
                feed_dict={
                    model.users: train_batch[:, 0],
                    model.pos_items: train_batch[:, 1],
                    model.scores1_p: np.expand_dims(train_batch[:, 4], 1),
                    model.scores1_n: np.expand_dims(train_batch[:, 5], 1),
                    model.items2: train_batch[:, 2],
                    model.labels2: np.expand_dims(train_batch[:, 3], 1),
                    model.scores2_p: np.expand_dims(train_batch[:, 6], 1),
                    model.scores2_n: np.expand_dims(train_batch[:, 7], 1),
                },
            )
        train_loss_list.append(loss)
        # calculate a test loss
        test_loss = sess.run(
            model.ideal_loss,
            feed_dict={
                model.users: test[:, 0],
                model.pos_items: test[:, 1],
                model.rel1: np.expand_dims(test[:, 3], 1),
                model.items2: test[:, 2],
                model.rel2: np.expand_dims(test[:, 4], 1),
            },
        )
        test_loss_list.append(test_loss)
    # calculate a validation loss
    if model_name in "bpr":
        val_loss = sess.run(
            model.unbiased_loss,
            feed_dict={
                model.users: val[:, 0],
                model.pos_items: val[:, 1],
                model.scores1: np.ones((num_val, 1)),
                model.items2: val[:, 2],
                model.labels2: np.zeros((num_val, 1)),
                model.scores2: np.ones((num_val, 1)),
            },
        )
    elif model_name in "ubpr":
        val_loss = sess.run(
            model.unbiased_loss,
            feed_dict={
                model.users: val[:, 0],
                model.pos_items: val[:, 1],
                model.scores1: np.expand_dims(val[:, 4], 1),
                model.items2: val[:, 2],
                model.labels2: np.expand_dims(val[:, 3], 1),
                model.scores2: np.expand_dims(val[:, 5], 1),
            },
        )
    elif model_name in "dubpr":
        val_loss = sess.run(
            model.unbiased_loss,
            feed_dict={
                model.users: val[:, 0],
                model.pos_items: val[:, 1],
                model.scores1_p: np.expand_dims(val[:, 4], 1),
                model.scores1_n: np.expand_dims(val[:, 5], 1),
                model.items2: val[:, 2],
                model.labels2: np.expand_dims(val[:, 3], 1),
                model.scores2_p: np.expand_dims(val[:, 6], 1),
                model.scores2_n: np.expand_dims(val[:, 7], 1),
            },
        )

    u_emb, i_emb = sess.run([model.user_embeddings, model.item_embeddings])
    if ~is_optuna:
        path = Path(f"../logs/{data}/{model_name}")
        (path / "loss").mkdir(parents=True, exist_ok=True)
        np.save(file=str(path / "loss/train.npy"), arr=train_loss_list)
        np.save(file=str(path / "loss/test.npy"), arr=test_loss_list)
        (path / "emb").mkdir(parents=True, exist_ok=True)
        np.save(file=str(path / "emb/user_embed.npy"), arr=u_emb)
        np.save(file=str(path / "emb/item_embed.npy"), arr=i_emb)
    sess.close()

    return u_emb, i_emb, val_loss


class Trainer:

    suffixes = ["cold-user", "rare-item"]
    at_k = [3, 5, 8]
    cold_user_threshold = 6
    rare_item_threshold = 100

    def __init__(
        self,
        data: str,
        pointwise_loss: str,
        pairwise_loss: str,
        max_iters: int = 1000,
        batch_size: int = 12,
        eta: float = 0.1,
        model_name: str = "bpr",
        propensity: str = 'original',
    ) -> None:
        """Initialize class."""
        self.data = data
        self.pointwise_loss = pointwise_loss
        self.pairwise_loss = pairwise_loss
        self.propensity = propensity

        if model_name not in ["expomf", "ip"]:
            hyper_params = yaml.safe_load(open(f"../conf/hyper_params.yaml", "r"))[
                data
            ][model_name]
            self.dim = np.int(hyper_params["dim"])
            self.lam = hyper_params["lam"]
            self.weight = hyper_params["weight"] if model_name == "wmf" else 1.0
            # TODO this is adjusted for dumf and ubpr
            # self.clip = hyper_params["clip"] if model_name in ["relmf", 'dumf'] else 0.0
            # self.beta = hyper_params["beta"] if model_name in ["ubpr", "dubpr"] else 0.0
            self.clip = hyper_params["clip"] if model_name == "relmf" else 0.0
            self.beta = hyper_params["beta"] if model_name == "ubpr" else 0.0
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eta = eta
        self.model_name = model_name

    def run(self, num_sims: int = 10) -> None:
        """Train implicit recommenders."""

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

        result_list = list()
        if self.data == "yahoo":
            cold_user_result_list = list()
            rare_item_result_list = list()
        for seed in np.arange(num_sims):
            tf.set_random_seed(12345) # TODO set seed to random 
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
                    loss_function=self.pairwise_loss
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
            elif self.model_name in ["wmf", "relmf", "dumf"]:
                point_rec = PointwiseRecommender(
                    num_users=num_users,
                    num_items=num_items,
                    weight=self.weight,
                    clip=self.clip,
                    dim=self.dim,
                    lam=self.lam,
                    eta=self.eta,
                    loss_function=self.pointwise_loss
                )
                u_emb, i_emb, _ = train_pointwise(
                    sess,
                    model=point_rec,
                    data=self.data,
                    train=train_point,
                    val=val_point,
                    test=test_point,
                    pscore=pscore,
                    nscore=nscore,
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
            elif self.model_name == "ip":
                u_emb, i_emb = train_ip(num_users=num_users, num_items=num_items, scores=pscore)

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

        ret_path = Path(f"../logs/{self.data}/{self.propensity}/{self.model_name}/results")
        ret_path.mkdir(parents=True, exist_ok=True)
        pd.concat(result_list, 1).to_csv(ret_path / f"aoa_all.csv")
        if self.data == "yahoo":
            pd.concat(cold_user_result_list, 1).to_csv(ret_path / f"aoa_cold-user.csv")
            pd.concat(rare_item_result_list, 1).to_csv(ret_path / f"aoa_rare-item.csv")

        results_all = pd.concat(result_list, 1)
        results_all["mean"] = results_all.mean(axis=1)
        print(results_all.head())
        print("Average DCG@3: ", results_all.loc["DCG@3":, "mean"])
