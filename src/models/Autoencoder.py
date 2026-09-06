import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self, n_features, bottleneck_dim, non_linear=False, hidden_dims_list=None, non_linear_function=nn.ReLU):
        super().__init__()

        encoder_layers = []
        in_dim = n_features
        for hidden_dim in hidden_dims_list:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            if non_linear:
                encoder_layers.append(non_linear_function())
            in_dim = hidden_dim

        decoder_layers = []
        in_dim = bottleneck_dim
        for hidden_dim in reversed(hidden_dims_list):
            decoder_layers.append(nn.Linear(in_dim, hidden_dim))
            if non_linear:
                decoder_layers.append(non_linear_function())
            in_dim = hidden_dim

        decoder_layers.append(nn.Linear(in_dim, n_features))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, X):
        z = self.encoder(X)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z