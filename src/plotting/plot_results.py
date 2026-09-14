import os
import matplotlib.pyplot as plt


def plot_reconstruction_results(
    pca_results,
    ae_results,
    vae_results,
    output_dir,
    filename="reconstruction_comparison.png"
):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))

    # PCA
    plt.plot(
        [result["dimension"] for result in pca_results],
        [result["test_reconstruction"] for result in pca_results],
        marker="o",
        label="PCA"
    )

    # Autoencoder
    plt.plot(
        [result["dimension"] for result in ae_results],
        [result["test_reconstruction"] for result in ae_results],
        marker="o",
        label="Nonlinear AE"
    )

    # VAE
    plt.plot(
        [result["dimension"] for result in vae_results],
        [result["test_reconstruction"] for result in vae_results],
        marker="o",
        label="Nonlinear VAE"
    )

    plt.xlabel("Bottleneck width (latent dimension)")
    plt.ylabel("Test reconstruction error (MSE)")

    plt.title(
        "Reconstruction Error Across Latent Dimensions"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(f"Plot saved to: {save_path}")

    return save_path


def plot_knn_results(
    pca_results,
    ae_results,
    vae_results,
    output_dir,
    filename="knn_comparison.png"
):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))

    # PCA
    plt.plot(
        [result["dimension"] for result in pca_results],
        [result["knn"] for result in pca_results],
        marker="o",
        label="PCA"
    )

    # Autoencoder
    plt.plot(
        [result["dimension"] for result in ae_results],
        [result["knn"] for result in ae_results],
        marker="o",
        label="Nonlinear AE"
    )

    # VAE
    plt.plot(
        [result["dimension"] for result in vae_results],
        [result["knn"] for result in vae_results],
        marker="o",
        label="Nonlinear VAE"
    )

    plt.xlabel("Bottleneck width (latent dimension)")
    plt.ylabel("k-NN accuracy")

    plt.title(
        "Classification Performance Across Latent Dimensions"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(f"Plot saved to: {save_path}")

    return save_path