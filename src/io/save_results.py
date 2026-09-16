from pathlib import Path
import pandas as pd

def save_or_update_results(dataset_name, method_name, results, results_dir="results"):
    dataset_dir = Path(results_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    file_path = dataset_dir / f"{method_name.lower()}_results.csv"

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
