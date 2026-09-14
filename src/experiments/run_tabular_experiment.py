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


activation_functions = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "Tanh": nn.Tanh,
}

def get_dataset_records():
    return merged_specs(Path("../databank/tabular_datasets.yaml"))



def execute_tabular(dataset_name: str, bottleneck_range: list, plot_path="./results"):

    specs = get_dataset_records() 
    if dataset_name not in list(specs):
        print(f"{dataset_name} doesn't exist...")
        raise
    spec = specs[dataset_name]

    # Step 1 - Fetch detaset from openml
    print("step 1/11 : Fetching the dataset..")
    dataset = fetch_openml(data_id=spec["data_id"], as_frame=True, parser="auto")

    X, y = dataset.data.copy(), dataset.target.copy()
    y = pd.Series(y, index=X.index)
    print("original Feature shape", X.shape, y.shape, type(X), type(y))

    print("step 2/11 : Cleaning data")

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

    print("step 3/11 : Splitting data")
    X_train, X_val, X_test, y_train, y_val, y_test, encoder = split_data(X, y)

    print("Step 4/11 : Preprocess data ")
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

    print("step 5/11 Tuning Autoencoder")
    study_ae = create_study_ae(
        X_train_tensor,
        X_val_tensor,
        n_epochs=100,
        bottleneck_dim=2,
        n_trials=20,
        study_name="autoencoder_HPO_mfeat_analysis"
    )
    print(f"Best parameters of Autencoder - {study_ae.best_params}")

    print("step 6/11 Tuning Variational Autoencoder")
    
    study_vae = create_study_vae(
        X_train_tensor,
        X_val_tensor,
        n_epochs=100,
        bottleneck_dim=32,
        n_trials=20,
        study_name="vae_hpo_study"
    )
    print(f"Best parameters of Variational Autencoder - {study_vae.best_params}")


    print("step 7/11 Finding best neighbors")
    best_neighbors, best_val_score = find_best_neighbors(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    print("Best neighbors:", best_neighbors)
    print("Validation accuracy:", best_val_score)

    input_dim = X_train_val_tensor.shape[1]

    print("step 8/11 Running PCA")

    pca_results = run_sweep(
        model_type="pca",
        dimensions=bottleneck_range,

        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test,
        y_test=y_test,

        n_neighbors=best_neighbors
    )

    print("step 9/11 Running AE")
    ae_results = run_sweep(
        model_type="ae",
        dimensions=bottleneck_range,
        input_dim=input_dim,
        X_train_val_tensor=X_train_val_tensor,
        X_test_tensor=X_test_tensor,
        y_train_val=y_train_val,
        y_test=y_test,
        best_params=study_ae.best_params,
        n_neighbors=best_neighbors,
        activation_functions=activation_functions,
        epochs=400
    )

    print("step 10/11 Running VAE")
    vae_results = run_sweep(
        model_type="vae",
        dimensions=bottleneck_range,
        input_dim=input_dim,
        X_train_val_tensor=X_train_val_tensor,
        X_test_tensor=X_test_tensor,
        y_train_val=y_train_val,
        y_test=y_test,
        best_params=study_vae.best_params,
        n_neighbors=best_neighbors,
        activation_functions=activation_functions,
        epochs=400
    )

    print("step 11/11 saving results")
    plot_reconstruction_results(
        pca_results=pca_results,
        ae_results=ae_results,
        vae_results=vae_results,
        output_dir=plot_path
    )
    plot_knn_results(
        pca_results=pca_results,
        ae_results=ae_results,
        vae_results=vae_results,
        output_dir = plot_path
    )



    


   