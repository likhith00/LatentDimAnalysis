import torch
import torch.nn as nn
import numpy as np
import optuna

from src.models.VAE import VariationalAutoencoder
from src.hpo.hidden_dims import build_hidden_dims_geo
from src.training.train_vae import train_vae
from src.utils import set_seed


def objective_vae(
    trial,
    X_train_tensor,
    X_val_tensor,
    n_epochs,
    bottleneck_dim
):
    seed = 42

    set_seed(seed)

    # =========================
    # Hyperparameters
    # =========================
    lr = trial.suggest_float( "lr", 1e-5, 1e-2, log=True)
    depth = trial.suggest_int("depth", 1, 3, step=1)
    activation_name = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "GELU", "Tanh"])
    beta = trial.suggest_float("beta",1e-4, 1.0, log=True)
    n_epochs = trial.suggest_int("n_epochs", 100, 500, step=50)

    activations = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "GELU": nn.GELU,
        "Tanh": nn.Tanh
    }

    non_linear_function = activations[activation_name]


    n_features = X_train_tensor.shape[1]
    input_dim = X_train_tensor.shape[1]

    hidden_dims_list = build_hidden_dims_geo(input_dim,latent_dim=bottleneck_dim, depth=depth)

    model = VariationalAutoencoder(
        n_features=n_features,
        bottleneck_dim=bottleneck_dim,
        hidden_dims_list=hidden_dims_list,
        non_linear_function=non_linear_function
    )

    loss = train_vae(
        model=model,
        X_train=X_train_tensor,
        X_val=X_val_tensor,
        trial=trial,
        n_epochs=n_epochs,
        lr=lr,
        beta=beta,
        patience=20
    )

    return loss


def create_study_vae(X_train_tensor, X_val_tensor, n_epochs, bottleneck_dim, n_trials=20, study_name=None):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
    )

    study.optimize(
        lambda trial: objective_vae(
            trial,
            X_train_tensor,
            X_val_tensor,
            n_epochs,
            bottleneck_dim,
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )

    return study