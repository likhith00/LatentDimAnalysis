import torch
import optuna
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def train_autoencoder(model, X_train, X_val, trial=None, n_epochs=200, lr=1e-3, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    model = model.to(device)
    X_train = X_train.to(device)
    X_val = X_val.to(device)


    for epoch in range(n_epochs):
        optimizer.zero_grad()

        x_hat, _ = model(X_train)
        loss = loss_fn(x_hat, X_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            x_val_hat, _ = model(X_val)
            val_loss = loss_fn(x_val_hat, X_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_loss


def train_final_model(final_model, X_train_val_tensor, X_test_tensor, best_params, epochs=None):
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
    loss_fn = nn.MSELoss()

    final_model = final_model.to(device)
    X_train_val_tensor = X_train_val_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    epochs = best_params["n_epochs"]

    final_model.train()
    for epoch in range(1,  epochs+ 1):
        optimizer.zero_grad()
        x_hat, _ = final_model(X_train_val_tensor)
        loss = loss_fn(x_hat, X_train_val_tensor)
        loss.backward()
        optimizer.step()

    final_model.eval()
    with torch.no_grad():
        x_train_val_hat, z_train_val = final_model(X_train_val_tensor)
        x_test_hat, z_test = final_model(X_test_tensor)
        test_loss = loss_fn(x_test_hat, X_test_tensor)

    return loss.item(), test_loss.item(), z_train_val, z_test