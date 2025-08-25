import torch
from torch import FloatTensor, IntTensor


def reduce_coordinates(coordinates: FloatTensor, indices: IntTensor) -> FloatTensor:
    return reduce_force(coordinates.view(-1), indices)


def reduce_force(forces_flattened: FloatTensor, indices: IntTensor) -> FloatTensor:
    """
    Delete specified rows from the force matrix.
    """
    indices = torch.unique(indices)

    N = forces_flattened.shape[0]
    device = forces_flattened.device
    keep_rows = torch.ones(N, dtype=torch.bool, device=device)
    keep_rows[indices] = False
    forces_flattened_reduced = forces_flattened[keep_rows]
    return forces_flattened_reduced


def reduce_hessian(Hessian: FloatTensor, indices: IntTensor) -> FloatTensor:
    """
    Delete specified rows and columns from the Hessian matrix.

    Args:
        hessian (FloatTensor): The Hessian matrix of shape (N, N) or (N*3, N*3).
        rows (IntTensor): 1D tensor of row indices to delete.
        cols (IntTensor): 1D tensor of column indices to delete.

    Returns:
        FloatTensor: The Hessian matrix with specified rows and columns removed.
    """
    # Sort and make unique for safety
    indices = torch.unique(indices)

    # Create masks for rows and columns to keep
    N = Hessian.shape[0]
    device = Hessian.device
    keep_rows = torch.ones(N, dtype=torch.bool, device=device)
    keep_cols = torch.ones(N, dtype=torch.bool, device=device)
    keep_rows[indices] = False
    keep_cols[indices] = False

    # Apply masks to delete rows and columns
    hessian_reduced = Hessian[keep_rows][:, keep_cols]
    return hessian_reduced