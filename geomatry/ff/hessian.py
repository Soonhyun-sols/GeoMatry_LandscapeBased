import torch
from torch import FloatTensor, IntTensor


@torch.jit.unused
def compute_hessians_vmap(
    forces: FloatTensor,
    positions: FloatTensor,
) -> FloatTensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -1 * forces_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(forces.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            I_N
        )[0]
    except RuntimeError:
        print("vmap failed, using loop")
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros((positions.shape[0], 3, positions.shape[0], 3))
    return gradient.reshape(num_elements, num_elements)


@torch.jit.unused
def compute_hessians_loop(
    forces: FloatTensor,
    positions: FloatTensor,
) -> FloatTensor:
    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hess_row = hess_row.detach()  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian


def reduce_hessian(hessian: FloatTensor, rows: IntTensor, cols: IntTensor) -> FloatTensor:
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
    rows = torch.unique(rows)
    cols = torch.unique(cols)

    # Create masks for rows and columns to keep
    N = hessian.shape[0]
    device = hessian.device
    keep_rows = torch.ones(N, dtype=torch.bool, device=device)
    keep_cols = torch.ones(N, dtype=torch.bool, device=device)
    keep_rows[rows] = False
    keep_cols[cols] = False

    # Apply masks to delete rows and columns
    hessian_reduced = hessian[keep_rows][:, keep_cols]
    return hessian_reduced