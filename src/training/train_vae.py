import torch
import optuna
import torch.nn as nn
import copy

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
    best_state = None
    patience_counter = 0

    model = model.to(device)
    X_train = X_train.to(device)
    X_val = X_val.to(device)

    warmup_epochs = 30
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        x_hat, mu, logvar, z = model(X_train)
        current_beta = beta * min(1.0, (epoch + 1) / warmup_epochs)

        loss, recon_loss, kl_loss = vae_loss(x_hat, X_train, mu, logvar, beta=current_beta)

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            mu_val, logvar_val = model.encode(X_val)
            x_val_hat = model.decode(mu_val)
            val_loss, val_recon_loss, val_kl_loss = vae_loss(
                x_val_hat,
                X_val,
                mu_val,
                logvar_val,
                beta=current_beta
            )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= patience:
                break

        if trial is not None:
            trial.report(val_loss.item(), epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

    if best_state is not None:
        model.load_state_dict(best_state)
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

    warmup_epochs = 30
    for epoch in range(1, epochs + 1):

        optimizer.zero_grad()

        x_hat, mu, logvar, z = final_model(
            X_train_val_tensor
        )

        current_beta = beta * min(1.0, epoch / warmup_epochs)
        loss, recon_loss, kl_loss = vae_loss(
            x_hat,
            X_train_val_tensor,
            mu,
            logvar,
            beta=current_beta
        )

        loss.backward()
        optimizer.step()

    final_model.eval()

    with torch.no_grad():


        mu_train_val, logvar_train_val = final_model.encode(
            X_train_val_tensor
        )

        mu_test, logvar_test = final_model.encode(
            X_test_tensor
        )

        x_train_val_hat = final_model.decode(mu_train_val)
        x_test_hat = final_model.decode(mu_test)

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
        test_recon_loss.item(),
        test_kl_loss.item(),
        mu_train_val,
        mu_test
    )