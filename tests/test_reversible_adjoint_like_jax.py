import os
import sys

import torch


# Ensure imports work whether pytest is run from repo root or from LearnedRegularizers/
HERE = os.path.dirname(__file__)
LR_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(LR_ROOT, ".."))
for p in (LR_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


from training_methods.reversible_deq import ReversibleSolver, solve_reversible_adjoint


class F(torch.nn.Module):
    """
    Small PyTorch analogue of the JAX test MLP:
      input: concat([z, x]) of size 2
      output: size 1
    """

    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 10, bias=True, dtype=dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 1, bias=True, dtype=dtype),
        )

    def forward(self, z, x):
        zx = torch.cat([z, x], dim=-1)
        return self.mlp(zx)


def grad_loss_rev(function, x, *, beta, tol=1e-10, max_steps=50):
    """
    PyTorch version of the JAX test's `grad_loss`: solve and return sum(z1).
    We include `x` in `params` to test gradients w.r.t. args as well.
    """
    z0 = torch.zeros(1, 1, dtype=x.dtype, device=x.device)

    def f(z, args):
        (x_,) = args
        return function(z, x_)

    params = [p for p in function.parameters() if p.requires_grad] + [x]
    z1, steps, err = solve_reversible_adjoint(
        f, z0, args=(x,), params=params, beta=beta, tol=tol, max_steps=max_steps
    )
    return z1.sum(), steps, err


def grad_loss_unroll(function, x, *, beta, tol=1e-10, max_steps=50):
    """
    Unrolled baseline: same forward iterations, but relying on standard autograd.
    """
    z0 = torch.zeros(1, 1, dtype=x.dtype, device=x.device)

    def f(z, args):
        (x_,) = args
        return function(z, x_)

    solver = ReversibleSolver(beta=beta)
    y, fz = solver.init(f, z0, (x,))
    z = z0.clone()
    steps_taken = 0
    err = float("inf")
    for i in range(max_steps):
        z, (y, fz), err = solver.step(f, z, (x,), (y, fz))
        steps_taken = i + 1
        if err < tol:
            break
    return z.sum(), steps_taken, err


@torch.no_grad()
def _zero_grads(module):
    for p in module.parameters():
        if p.grad is not None:
            p.grad.zero_()


def _collect_param_grads(module):
    grads = []
    for p in module.parameters():
        grads.append(None if p.grad is None else p.grad.detach().clone())
    return grads


def _assert_close(a, b, *, rtol, atol, msg):
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=msg)


def test_reversible_adjoint_like_jax(beta=0.8):
    # Mirror JAX test intent:
    # grads(reversible adjoint) == grads(checkpointed/unrolled) for same solver
    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"

    function = F(dtype=dtype).to(device)
    x = torch.tensor([[0.1]], dtype=dtype, device=device, requires_grad=True)

    # --- Unrolled ---
    _zero_grads(function)
    if x.grad is not None:
        x.grad.zero_()
    loss_u, steps_u, err_u = grad_loss_unroll(function, x, beta=beta, tol=1e-12, max_steps=80)
    loss_u.backward()
    grads_u = _collect_param_grads(function)
    gradx_u = x.grad.detach().clone()

    # --- Reversible adjoint ---
    _zero_grads(function)
    x.grad.zero_()
    loss_r, steps_r, err_r = grad_loss_rev(function, x, beta=beta, tol=1e-12, max_steps=80)
    loss_r.backward()
    grads_r = _collect_param_grads(function)
    gradx_r = x.grad.detach().clone()

    assert steps_u == steps_r, f"steps mismatch: unroll={steps_u}, rev={steps_r}"
    _assert_close(torch.tensor(err_u), torch.tensor(err_r), rtol=1e-8, atol=1e-10, msg="err mismatch")
    _assert_close(loss_u.detach(), loss_r.detach(), rtol=1e-8, atol=1e-10, msg="loss mismatch")

    # Parameter gradients
    for i, (gu, gr) in enumerate(zip(grads_u, grads_r)):
        assert gu is not None and gr is not None
        _assert_close(gu, gr, rtol=1e-3, atol=1e-8, msg=f"param grad mismatch at idx {i}")

    # Args gradient (x)
    _assert_close(gradx_u, gradx_r, rtol=1e-3, atol=1e-8, msg="x grad mismatch")


def test_reversible_adjoint_like_jax_beta_08():
    test_reversible_adjoint_like_jax(beta=0.8)


def test_reversible_adjoint_like_jax_beta_12():
    # JAX tests include beta=1.2; this may not converge, but we compare against the same unrolled steps.
    test_reversible_adjoint_like_jax(beta=1.2)

