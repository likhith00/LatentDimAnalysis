from pathlib import Path
import pandas as pd
import json

def save_or_update_results(dataset_name, method_name, results, results_dir="results"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    file_path = results_dir / f"{method_name.lower()}_results.csv"

    new_df = pd.DataFrame(results)

    if not file_path.exists():
        new_df = new_df.sort_values("dimension")
        new_df.to_csv(file_path, index=False)
        print(f"Created: {file_path}")
        return

    old_df = pd.read_csv(file_path)

    old_df = old_df[
        ~old_df["dimension"].isin(new_df["dimension"])
    ]

    combined_df = pd.concat(
        [old_df, new_df],
        ignore_index=True
    )

    combined_df = combined_df.sort_values("dimension").reset_index(drop=True)

    combined_df.to_csv(file_path, index=False)

    print(f"Updated: {file_path}")

def load_results(dataset_name, results_dir = "results"):
    results_dir = Path(results_dir)
    pca_path = results_dir / "pca_results.csv"
    ae_path = results_dir / "ae_results.csv"
    vae_path = results_dir / "vae_results.csv"

    pca_results = pd.read_csv(pca_path).to_dict("records")
    ae_results = pd.read_csv(ae_path).to_dict("records")
    vae_results = pd.read_csv(vae_path).to_dict("records")

    return pca_results, ae_results, vae_results

def load_hpo_params(dataset_name, method_name, results_dir="results"):
    results_dir = Path(results_dir)
    filename = f"{method_name.lower()}_hpo.json"
    hpo_file = results_dir / filename

    if not hpo_file.exists():
        hpo_file = results_dir / dataset_name / filename

    if not hpo_file.exists():
        return None

    with open(hpo_file, "r") as f:
        params = json.load(f)

    print(f"Loaded HPO parameters from: {hpo_file}")

    return params

def save_hpo_params(dataset_name, method_name, best_params, results_dir="results"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    hpo_file = results_dir / f"{method_name.lower()}_hpo.json"

    with open(hpo_file, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Saved HPO parameters to: {hpo_file}")