import torch
import torch.nn as nn
import numpy as np
from src.models.Autoencoder import Autoencoder
from src.hpo.hidden_dims import build_hidden_dims_static
from src.training.train import train_autoencoder
import optuna

def objective(trial, X_train_tensor, X_val_tensor,n_epochs, bottleneck_dim):

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    depth = trial.suggest_int("depth", 1,3, step=1)
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'GELU', 'Tanh'])
    activations = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'GELU': nn.GELU,
        'Tanh': nn.Tanh
    }

    non_linear_function = activations[activation_name]

    n_features = X_train_tensor.shape[1]
    input_dim = X_train_tensor.shape[1]


    #hidden_dims_list = build_hidden_dims_static(input_dim, bottleneck_dim, depth)
    hidden_dims_list = build_hidden_dims_static(input_dim, depth=depth)

    model = Autoencoder(
        n_features=n_features,
        bottleneck_dim=bottleneck_dim,
        non_linear=True,
        hidden_dims_list=hidden_dims_list,
        non_linear_function=non_linear_function
    )
    loss = train_autoencoder(
        model=model,
        X_train=X_train_tensor,
        trial=trial,
        X_val=X_val_tensor,
        n_epochs=n_epochs,
        lr=lr,
        patience=20
    )
    return loss

def create_study(
    X_train_tensor,
    X_val_tensor,
    n_epochs,
    bottleneck_dim,
    n_trials=20,
    study_name=None,
):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
    )

    study.optimize(
        lambda trial: objective(
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