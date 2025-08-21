import torch
from torch import FloatTensor


@torch.jit.unused
def compute_hessians_vmap(
    Fa: FloatTensor,
    Ra: FloatTensor,
) -> FloatTensor:
    forces_flatten = Fa.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -1 * forces_flatten,
            Ra,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(Fa.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            I_N
        )[0]
    except RuntimeError:
        print("vmap failed, using loop")
        gradient = compute_hessians_loop(Fa, Ra)
    if gradient is None:
        return torch.zeros((Ra.shape[0], 3, Ra.shape[0], 3))
    return gradient.reshape(num_elements, num_elements)


@torch.jit.unused
def compute_hessians_loop(
    Fa: FloatTensor,
    Ra: FloatTensor,
) -> FloatTensor:
    hessian = []
    for grad_elem in Fa.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[Ra],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hess_row = hess_row.detach()  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(Ra))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian
