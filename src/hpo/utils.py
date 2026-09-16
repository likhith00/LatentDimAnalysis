from src.io.save_results import load_hpo_params, save_hpo_params


def get_or_run_hpo(dataset_name, method_name, run_hpo_function, results_dir="results"):
    best_params = load_hpo_params(
        dataset_name=dataset_name,
        method_name=method_name,
        results_dir=results_dir
    )

    if best_params is not None:
        print(f"Skipping {method_name} HPO.")
        return best_params

    print(f"No saved {method_name} HPO found. Running HPO...")

    best_params = run_hpo_function()

    save_hpo_params(
        dataset_name=dataset_name,
        method_name=method_name,
        best_params=best_params,
        results_dir=results_dir
    )

    return best_params