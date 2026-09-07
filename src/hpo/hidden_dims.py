import numpy as np


def build_hidden_dims_geo(input_dim, latent_dim, depth):
    values = np.geomspace(input_dim, latent_dim, num=depth+2)[1:-1]
    hidden_dims = []
    previous = input_dim 
    for value in values:
        width = int(round(value))
        width = max(latent_dim + 1, width)
        width = min(previous - 1, width)

        if width <= latent_dim:
            break
        hidden_dims.append(width)
        previous = width
    return hidden_dims

def build_hidden_dims_static(input_dim, depth=1):
    if depth == 1:
        return [2 * input_dim]

    if depth == 2:
        return [2 * input_dim, input_dim]

    if depth == 3:
        return [2 * input_dim, input_dim, input_dim]

    raise ValueError("Unsupported depth")