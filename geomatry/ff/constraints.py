import torch
from torch import FloatTensor, IntTensor


def reduce_force(Fa: FloatTensor, indices: IntTensor) -> FloatTensor:
    """
    Delete specified rows from the force matrix.
    """
    indices = torch.unique(indices)
    N = Fa.shape[0]
    device = Fa.device
    keep_rows = torch.ones(N, dtype=torch.bool, device=device)
    keep_rows[indices] = False
    Fa_reduced = Fa[keep_rows]
    return Fa_reduced


def reduce_hessian(Haa: FloatTensor, indices: IntTensor) -> FloatTensor:
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
    N = Haa.shape[0]
    device = Haa.device
    keep_rows = torch.ones(N, dtype=torch.bool, device=device)
    keep_cols = torch.ones(N, dtype=torch.bool, device=device)
    keep_rows[indices] = False
    keep_cols[indices] = False

    # Apply masks to delete rows and columns
    hessian_reduced = Haa[keep_rows][:, keep_cols]
    return hessian_reduced