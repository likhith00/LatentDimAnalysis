from pathlib import Path

from torch import nn

from src.data.image import preprocess_torchvision_dataset
from src.experiments.evaluate_dimensions import run_sweep
from src.hpo.autoencoder_hpo import create_study_ae
from src.hpo.vae_autoencoder_hpo import create_study_vae
from src.io.read_file import merged_specs
from src.load.image import create_image_transform, get_dataset_mapping, load_dataset_tvt
import torch
from torch.utils.data import ConcatDataset

from src.metrics.knn import find_best_neighbors
from src.plotting.plot_results import plot_knn_results, plot_reconstruction_results
from src.utils import aggregate_seed_results
from src.io.save_results import load_hpo_params, save_hpo_params, save_or_update_results, load_results

activation_functions = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "Tanh": nn.Tanh,
}


def get_dataset_records():
    return merged_specs(Path("../databank/image_datasets.yaml"))

def execute_image(dataset_name: str, bottleneck_range: list, transform:bool = None, hpo_latent:int=32, seeds=[42], plot_path="./results" ):
    specs = get_dataset_records()

    if dataset_name not in list(specs):
            print(f"{dataset_name} doesn't exist...")
            raise
    spec = specs[dataset_name]
    
    print(f"Dataset: {spec['display_name']}")
    print(f"Expected classes: {spec['expected_classes']}")
    print(f"Samples: {spec['samples_to_load']}")

    print("step 1/10 : Fetching the dataset..")

    dataset_mapping = get_dataset_mapping(dataset_name=dataset_name)
    trf = None
    results_dir = "results"


    if transform:
        trf = create_image_transform(
            spec["size"]
        )

    train_dataset, val_dataset, test_dataset = load_dataset_tvt(
        name=dataset_name,
        dataset_mapping=dataset_mapping,
        root="../root",
        transform=trf
    )

    full_dataset = ConcatDataset([
        train_dataset,
        val_dataset,
        test_dataset
    ])

    print("step 2/10 : preprocessing and splitting the dataset..")
    prep_data = preprocess_torchvision_dataset(
        dataset=full_dataset,
        max_samples=spec["samples_to_load"],
        batch_size=128,
        num_workers=4
    )

    X_train, X_test,X_val = prep_data["X_train"], prep_data["X_test"], prep_data["X_val"]
    y_train, y_test, y_val = prep_data["y_train"], prep_data["y_test"], prep_data["y_val"]

    X_train_tensor, X_test_tensor, X_val_tensor = prep_data["X_train_tensor"],prep_data["X_test_tensor"], prep_data["X_val_tensor"]
    y_train_tensor, y_test_tensor, y_val_tensor = prep_data["y_train_tensor"], prep_data["y_test_tensor"], prep_data["y_val_tensor"]

    X_train_val_tensor = torch.cat([X_train_tensor, X_val_tensor], dim=0)
    y_train_val_tensor = torch.cat([y_train_tensor, y_val_tensor], dim=0)
    X_train_val = X_train_val_tensor.detach().cpu().numpy()
    y_train_val = y_train_val_tensor.detach().cpu().numpy()

    print("split shape: X_train, X_val, X_test, X_train_val ",X_train.shape, X_val.shape, X_test.shape, X_train_val.shape)

    print(f"step 3/10 Tuning Autoencoder for the latent dimension: {hpo_latent}")
    best_ae_params = load_hpo_params( dataset_name=dataset_name, method_name="AE", results_dir=results_dir)

    if best_ae_params is None:
        print("No existing AE HPO found. ""Running Optuna...")

        study_ae = create_study_ae(
            X_train_tensor,
            X_val_tensor,
            n_epochs=100,
            bottleneck_dim=hpo_latent,
            n_trials=20,
            study_name=f"{dataset_name}_ae_hpo"
        )

        best_ae_params = study_ae.best_params.copy()

        best_ae_params["hpo_latent"] = hpo_latent
        save_hpo_params(
            dataset_name=dataset_name,
            method_name="AE",
            best_params=best_ae_params,
            results_dir=results_dir
        )
    else:
        print("Existing AE HPO found. Skipping HPO.")

    print("Best parameters of Autoencoder:", best_ae_params)

    print("step 4/10 Tuning Variational Autoencoder")
    
    best_vae_params = load_hpo_params(dataset_name=dataset_name, method_name="VAE", results_dir=results_dir)
    if best_vae_params is None:
        print("No existing VAE HPO found. " "Running Optuna...")

        study_vae = create_study_vae(
            X_train_tensor,
            X_val_tensor,
            n_epochs=100,
            bottleneck_dim=hpo_latent,
            n_trials=20,
            study_name=f"{dataset_name}_vae_hpo"
        )

        best_vae_params = study_vae.best_params.copy()

        best_vae_params["hpo_latent"] = hpo_latent

        save_hpo_params(
            dataset_name=dataset_name,
            method_name="VAE",
            best_params=best_vae_params,
            results_dir=results_dir
        )
    else:
        print("Existing VAE HPO found. Skipping HPO.")
    print("Best parameters of Variational Autoencoder:",best_vae_params)


    print("step 5/10 Finding best neighbors")
    best_neighbors, best_val_score = find_best_neighbors(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    print("Best neighbors:", best_neighbors)
    print("Validation accuracy:", best_val_score)

    input_dim = X_train_val_tensor.shape[1]

    print("step 6/10 Running PCA")

    pca_raw_results = run_sweep(
            model_type="pca",
            dimensions=bottleneck_range,
    
            X_train_val=X_train_val,
            y_train_val=y_train_val,
            X_test=X_test,
            y_test=y_test,
    
            n_neighbors=best_neighbors
        )

    print("step 7/10 Running AE")

    ae_raw_results = run_sweep(
        model_type="ae",
        dimensions=bottleneck_range,
        seeds=seeds,
        input_dim=input_dim,
        X_train_val_tensor=X_train_val_tensor,
        X_test_tensor=X_test_tensor,
        y_train_val=y_train_val,
        y_test=y_test,
        best_params=best_ae_params,
        n_neighbors=best_neighbors,
        activation_functions=activation_functions,
        epochs=400
    )

    print("step 8/10 Running VAE")

    vae_raw_results = run_sweep(
        model_type="vae",
        dimensions=bottleneck_range,
        seeds=seeds,
        input_dim=input_dim,
        X_train_val_tensor=X_train_val_tensor,
        X_test_tensor=X_test_tensor,
        y_train_val=y_train_val,
        y_test=y_test,
        best_params=best_vae_params,
        n_neighbors=best_neighbors,
        activation_functions=activation_functions,
        epochs=400
    )

    print("Step 9/10 Aggregating seed results...")

    pca_results = aggregate_seed_results(
        pca_raw_results
    )

    ae_results = aggregate_seed_results(
        ae_raw_results
    )

    vae_results = aggregate_seed_results(
        vae_raw_results
    )

    print("step 10/10 Plotting and saving results")
    save_or_update_results(
        dataset_name=dataset_name,
        method_name="PCA",
        results=pca_results,
        results_dir="results"
    )

    save_or_update_results(
        dataset_name=dataset_name,
        method_name="AE",
        results=ae_results,
        results_dir="results"
    )

    save_or_update_results(
        dataset_name=dataset_name,
        method_name="VAE",
        results=vae_results,
        results_dir="results"
    )    

    all_pca_results, all_ae_results, all_vae_results = load_results(
        dataset_name=dataset_name,
        results_dir="results"
    )
    
    plot_reconstruction_results(
        pca_results=all_pca_results,
        ae_results=all_ae_results,
        vae_results=all_vae_results,
        output_dir=plot_path
    )
    plot_knn_results(
        pca_results=all_pca_results,
        ae_results=all_ae_results,
        vae_results=all_vae_results,
        output_dir = plot_path
    )
    
    