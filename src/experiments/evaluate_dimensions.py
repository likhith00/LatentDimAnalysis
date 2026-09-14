import torch 
import numpy as np
from sklearn.decomposition import PCA
from src.models.VAE import VariationalAutoencoder
from src.models.AE import Autoencoder
from src.training.train_vae import train_final_model_vae
from src.training.train import train_final_model
from src.metrics.knn import compute_knn_score
from src.hpo.hidden_dims import build_hidden_dims_geo



def evaluate_pca_dimension(
    k,
    X_train_val,
    y_train_val,
    X_test,
    y_test,
    n_neighbors
):
    pca = PCA(n_components=k)

    X_train_latent = pca.fit_transform(X_train_val)
    X_test_latent = pca.transform(X_test)

    X_train_recon = pca.inverse_transform(X_train_latent)
    X_test_recon = pca.inverse_transform(X_test_latent)

    train_reconstruction_error = np.mean(
        (X_train_val - X_train_recon) ** 2
    )

    test_reconstruction_error = np.mean(
        (X_test - X_test_recon) ** 2
    )

    knn_accuracy = compute_knn_score(
        X_train=X_train_latent,
        y_train=y_train_val,
        X_test=X_test_latent,
        y_test=y_test,
        n_neighbors=n_neighbors
    )

    return {
        "dimension": k,
        "knn": knn_accuracy,
        "train_reconstruction": train_reconstruction_error,
        "test_reconstruction": test_reconstruction_error
    }

def evaluate_ae_dimension(
    k,
    input_dim,
    X_train_val_tensor,
    X_test_tensor,
    y_train_val,
    y_test,
    best_params,
    n_neighbors,
    activation_functions,
    epochs=400,
    seed=42
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_dims = build_hidden_dims_geo(
        input_dim=input_dim,
        latent_dim=k,
        depth=best_params["depth"]
    )

    model = Autoencoder(
        n_features=input_dim,
        bottleneck_dim=k,
        hidden_dims_list=hidden_dims,
        non_linear=True,
        non_linear_function=activation_functions[
            best_params["activation"]
        ]
    )

    (
        train_loss,
        test_loss,
        z_train,
        z_test
    ) = train_final_model(
        final_model=model,
        X_train_val_tensor=X_train_val_tensor,
        X_test_tensor=X_test_tensor,
        best_params=best_params,
        epochs=epochs
    )

    z_train = z_train.cpu().numpy()
    z_test = z_test.cpu().numpy()

    knn_accuracy = compute_knn_score(
        X_train=z_train,
        y_train=y_train_val,
        X_test=z_test,
        y_test=y_test,
        n_neighbors=n_neighbors
    )

    return {
        "dimension": k,
        "knn": knn_accuracy,
        "train_reconstruction": train_loss,
        "test_reconstruction": test_loss
    }

def evaluate_vae_dimension(
    k,
    input_dim,
    X_train_val_tensor,
    X_test_tensor,
    y_train_val,
    y_test,
    best_params,
    n_neighbors,
    activation_functions,
    epochs=400,
    seed=42
):

    if k >= input_dim:
        raise ValueError(
            f"Latent dimension {k} must be smaller than input dimension {input_dim}"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    hidden_dims = build_hidden_dims_geo(
        input_dim=input_dim,
        latent_dim=k,
        depth=best_params["depth"]
    )

    model = VariationalAutoencoder(
        n_features=input_dim,
        bottleneck_dim=k,
        hidden_dims_list=hidden_dims,
        non_linear_function=activation_functions[
            best_params["activation"]
        ]
    )

    (
        train_total_loss,
        test_total_loss,
        test_reconstruction,
        test_kl,
        mu_train,
        mu_test
    ) = train_final_model_vae(
        final_model=model,
        X_train_val_tensor=X_train_val_tensor,
        X_test_tensor=X_test_tensor,
        best_params=best_params,
        epochs=epochs
    )

    mu_train = mu_train.detach().cpu().numpy()
    mu_test = mu_test.detach().cpu().numpy()

    knn_accuracy = compute_knn_score(
        X_train=mu_train,
        y_train=y_train_val,
        X_test=mu_test,
        y_test=y_test,
        n_neighbors=n_neighbors
    )

    return {
        "dimension": k,
        "knn": knn_accuracy,
        "train_total_loss": train_total_loss,
        "test_total_loss": test_total_loss,
        "test_reconstruction": test_reconstruction,
        "test_kl": test_kl
    }
      
def run_sweep(
    model_type,
    dimensions,
    **kwargs
):

    results = []

    for k in dimensions:

        if model_type == "pca":
            result = evaluate_pca_dimension(
                k=k,
                **kwargs
            )

        elif model_type == "ae":
            result = evaluate_ae_dimension(
                k=k,
                **kwargs
            )

        elif model_type == "vae":
            result = evaluate_vae_dimension(
                k=k,
                **kwargs
            )

        else:
            raise ValueError(
                f"Unknown model type: {model_type}"
            )

        results.append(result)

        print(result)

    return results