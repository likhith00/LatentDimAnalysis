import random
import numpy as np
import torch

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # More deterministic CUDA behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import numpy as np


def aggregate_seed_results(results):
    aggregated = []

    dimensions = sorted(
        set(result["dimension"] for result in results)
    )

    for dimension in dimensions:
        dimension_results = [
            result
            for result in results
            if result["dimension"] == dimension
        ]
        aggregated_result = {
            "dimension": dimension
        }
        keys = [
            key
            for key in dimension_results[0].keys()
            if key not in ["dimension", "seed"]
        ]

        for key in keys:

            values = [
                result[key]
                for result in dimension_results
            ]

            aggregated_result[key] = np.mean(values)
            aggregated_result[f"{key}_std"] = np.std(values)

        aggregated.append(aggregated_result)

    return aggregated