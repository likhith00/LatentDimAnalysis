from pathlib import Path
import torch 
import pandas as pd
from sklearn.datasets import fetch_openml
from src.io.read_file import merged_specs
from src.data.tabular import (
    get_num_cat_columns,
    preprocess_features,
    remove_constant_columns,
    remove_duplicate_rows,
    remove_identifier_columns,
    remove_missing_label,
    split_data,
)
import torch.nn as nn
from src.hpo.vae_autoencoder_hpo import create_study_vae
from src.hpo.autoencoder_hpo import create_study_ae
from src.metrics.knn import find_best_neighbors
from src.experiments.evaluate_dimensions import run_sweep
from src.plotting.plot_results import (
    plot_knn_results,
    plot_reconstruction_results
)
from src.utils import aggregate_seed_results
from src.io.save_results import load_hpo_params, save_hpo_params, save_or_update_results, load_results


activation_functions = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "Tanh": nn.Tanh,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_dataset_records():
    return merged_specs(PROJECT_ROOT / "databank" / "tabular_datasets.yaml")



def execute_tabular(dataset_name: str, bottleneck_range: list, hpo_latent=32, seeds=[42], results_path = "results"):
    specs = get_dataset_records() 
    if dataset_name not in list(specs):
        print(f"{dataset_name} doesn't exist...")
        raise
    spec = specs[dataset_name]
    main_results_dir = Path(results_path)
    if not main_results_dir.is_absolute():
        main_results_dir = PROJECT_ROOT / main_results_dir
    hpo_results_dir =  main_results_dir / dataset_name /"hpo"
    model_results_dir = main_results_dir / dataset_name/ "model_results"
    plots_dir = main_results_dir / dataset_name / "plots"

    # Step 1 - Fetch detaset from openml
    print("step 1/12 : Fetching the dataset..")
    dataset = fetch_openml(data_id=spec["data_id"], as_frame=True, parser="auto")

    X, y = dataset.data.copy(), dataset.target.copy()
    y = pd.Series(y, index=X.index)
    print("original Feature shape", X.shape, y.shape, type(X), type(y))

    print("step 2/12 : Cleaning data")

    X, y, duplicate_count = remove_duplicate_rows(X, y)
    print(f"No.of duplicate rows removed: {duplicate_count}")

    X, y, missing_label_count = remove_missing_label(X, y)
    print(f"No.of duplicate rows with missing labels removed: {missing_label_count}")

    id_columns = spec.get("identifier_columns", [])
    X = remove_identifier_columns(X, spec.get("identifier_columns", []))
    print(f"No.of identifier columns removed: {len(id_columns)}")

    X, constant_columns = remove_constant_columns(X)
    print(f"No.of constant columns removed: {len(constant_columns)}")

    X, numeric, categorical = get_num_cat_columns(
        X,
        categorical_columns=spec.get("manual_categorical_columns", []),
        numeric_columns=spec.get("manual_numeric_columns", []),
    )
    print(f"No. of categorical columns: {len(numeric)} and No.of numerical columns: {len(categorical)}")

    print("step 3/12 : Splitting data")
    X_train, X_val, X_test, y_train, y_val, y_test, encoder = split_data(X, y)

    print("Step 4/12 : Preprocess data ")
    X_train, X_val, X_test = preprocess_features(
        X_train, X_val, X_test, numeric, categorical
    )

    
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train)

    X_val_tensor   = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val)

    X_test_tensor  = torch.from_numpy(X_test).float()
    X_train_val_tensor = torch.cat([X_train_tensor, X_val_tensor], dim=0)
    y_train_val_tensor = torch.cat([y_train_tensor, y_val_tensor], dim=0)

    X_train_val = X_train_val_tensor.detach().cpu().numpy()
    y_train_val = y_train_val_tensor.detach().cpu().numpy()

    print("split shape: X_train, X_val, X_test, X_train_val ",X_train.shape, X_val.shape, X_test.shape, X_train_val.shape)

    print("step 5/12 Tuning Autoencoder")

    best_ae_params = load_hpo_params( dataset_name=dataset_name, method_name="ae", results_dir=hpo_results_dir)
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
            results_dir=hpo_results_dir
        )
    else:
        print("Existing AE HPO found. Skipping HPO.")

    print("Best parameters of Autoencoder:", best_ae_params)

    print("step 6/12 Tuning Variational Autoencoder")
    
    best_vae_params = load_hpo_params(dataset_name=dataset_name, method_name="vae", results_dir=hpo_results_dir)
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
            results_dir=hpo_results_dir
        )
    else:
        print("Existing VAE HPO found. Skipping HPO.")
    print("Best parameters of Variational Autoencoder:",best_vae_params)



    print("step 7/12 Finding best neighbors")
    best_neighbors, best_val_score = find_best_neighbors(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    print("Best neighbors:", best_neighbors)
    print("Validation accuracy:", best_val_score)

    input_dim = X_train_val_tensor.shape[1]

    print("step 8/12 Running PCA")

    pca_raw_results = run_sweep(
        model_type="pca",
        dimensions=bottleneck_range,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test,
        y_test=y_test,

        n_neighbors=best_neighbors
    )

    print("step 9/12 Running AE")
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
    

    print("step 10/12 Running VAE")
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

    print("step 11/12 Aggregating seed results...")
    pca_results = aggregate_seed_results(pca_raw_results)
    ae_results = aggregate_seed_results(ae_raw_results)
    vae_results = aggregate_seed_results(vae_raw_results)

    print("step 12/12 saving results and  plots")
    save_or_update_results(
        dataset_name=dataset_name,
        method_name="PCA",
        results=pca_results,
        results_dir=model_results_dir
    )

    save_or_update_results(
        dataset_name=dataset_name,
        method_name="AE",
        results=ae_results,
        results_dir=model_results_dir
    )

    save_or_update_results(
        dataset_name=dataset_name,
        method_name="VAE",
        results=vae_results,
        results_dir=model_results_dir
    )

    all_pca_results, all_ae_results, all_vae_results = load_results(
        dataset_name=dataset_name,
        results_dir=model_results_dir
    )
    
    plot_reconstruction_results(
        pca_results=all_pca_results,
        ae_results=all_ae_results,
        vae_results=all_vae_results,
        output_dir=plots_dir
    )
    plot_knn_results(
        pca_results=all_pca_results,
        ae_results=all_ae_results,
        vae_results=all_vae_results,
        output_dir = plots_dir
    )


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tabular latent-dimension experiments.")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name as defined in tabular_datasets.yaml"
    )
    parser.add_argument(
        "--bottleneck-range",
        type=int,
        nargs="+",
        required=True,
        help="Latent dimensions to evaluate, e.g. --bottleneck-range 1 2 4 8 16"
    )
    parser.add_argument(
        "--hpo-latent",
        type=int,
        default=32,
        help="Reference latent dimension used for HPO"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds, e.g. --seeds 42 43 44"
    )

    parser.add_argument(
        "--results-path",
        type=str,
        default="results",
        help="Root directory for saved results"
    )

    args = parser.parse_args()

    execute_tabular(
        dataset_name=args.dataset,
        bottleneck_range=args.bottleneck_range,
        hpo_latent=args.hpo_latent,
        seeds=args.seeds,
        results_path=args.results_path
    )


   