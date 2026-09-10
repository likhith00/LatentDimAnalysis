import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, n_features, bottleneck_dim, hidden_dims_list, non_linear_function=nn.ReLU):
        super().__init__()

        encoder_layers = []
        in_dim = n_features

        for hidden_dim in hidden_dims_list:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            encoder_layers.append(non_linear_function())
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(in_dim, bottleneck_dim)
        self.fc_logvar = nn.Linear(in_dim, bottleneck_dim)

        decoder_layers = []
        in_dim = bottleneck_dim

        for hidden_dim in reversed(hidden_dims_list):
            decoder_layers.append(nn.Linear(in_dim, hidden_dim))
            decoder_layers.append(non_linear_function())
            in_dim = hidden_dim

        decoder_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar, z