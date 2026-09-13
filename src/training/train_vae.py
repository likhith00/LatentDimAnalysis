import torch
import optuna
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def vae_loss(x_hat, x, mu, logvar, beta=1.0):

    reconstruction_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss


def train_vae(model, X_train, X_val, trial=None, n_epochs=200, lr=1e-3, beta=1.0, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    model = model.to(device)
    X_train = X_train.to(device)
    X_val = X_val.to(device)


    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        x_hat, mu, logvar, z = model(X_train)

        loss, recon_loss, kl_loss = vae_loss(x_hat, X_train, mu, logvar, beta=beta)

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            x_val_hat, mu_val, logvar_val, z_val = model(X_val)

            val_loss, val_recon_loss, val_kl_loss = vae_loss(
                x_val_hat,
                X_val,
                mu_val,
                logvar_val,
                beta=beta
            )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= patience:
                break

        if trial is not None:
            trial.report(val_loss.item(), epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_loss


def train_final_model_vae(final_model, X_train_val_tensor, X_test_tensor, best_params, epochs=None):

    final_model = final_model.to(device)

    X_train_val_tensor = X_train_val_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)

    lr = best_params["lr"]
    beta = best_params.get("beta", 1.0)
    if epochs is None:
        epochs = best_params["n_epochs"]

    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=lr
    )

    final_model.train()

    for epoch in range(1, epochs + 1):

        optimizer.zero_grad()

        x_hat, mu, logvar, z = final_model(X_train_val_tensor)

        loss, recon_loss, kl_loss = vae_loss(
            x_hat,
            X_train_val_tensor,
            mu,
            logvar,
            beta=beta
        )

        loss.backward()
        optimizer.step()

    final_model.eval()

    with torch.no_grad():

        x_train_val_hat, mu_train_val, logvar_train_val, z_train_val = final_model(X_train_val_tensor)

        # Test latent representation
        x_test_hat, mu_test, logvar_test, z_test = final_model(X_test_tensor)

        test_loss, test_recon_loss, test_kl_loss = vae_loss(
            x_test_hat,
            X_test_tensor,
            mu_test,
            logvar_test,
            beta=beta
        )

    return (
        loss.item(),
        test_loss.item(),
        recon_loss.item(),
        kl_loss.item(),
        z_train_val,
        z_test
    )