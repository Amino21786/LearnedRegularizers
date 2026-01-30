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


from training_methods.reversible_deq import solve_reversible_adjoint, ReversibleSolver


def _unroll_reversible(function, z0, args, beta=0.8, tol=1e-6, max_steps=5):
    """
    Unrolled version of the reversible forward solver that builds a normal autograd graph.
    We mirror the algorithm used in ReversibleSolver (matching the JAX reversible solver):
      - keep (y, fz) state where fz = f(z)
      - y <- (1-beta)*y + beta*fz
      - z <- (1-beta)*z + beta*f(y)
      - fz <- f(z)
    """
    solver = ReversibleSolver(beta=beta)
    y, fz = solver.init(function, z0, args)
    z = z0.clone()
    steps_taken = 0
    error = float("inf")

    for i in range(max_steps):
        z, (y, fz), error = solver.step(function, z, args, (y, fz))
        steps_taken = i + 1
        # Match the forward implementation behavior: early stop by numeric residual
        if error < tol:
            break

    return z, steps_taken, error


def _assert_allclose(a, b, rtol=1e-4, atol=1e-6, msg=""):
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=msg)


def test_revdeq_adjoint_matches_unroll_linear():
    torch.manual_seed(0)
    device = "cpu"

    # Simple contraction linear map f(z) = z @ W
    W = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32, device=device))

    def f(z, args):
        (W_,) = args
        return z @ W_

    z0 = torch.ones(1, 1, dtype=torch.float32, device=device)
    beta = 0.8
    tol = 1e-6
    max_steps = 5

    # Unrolled baseline
    W.grad = None
    z1_u, steps_u, err_u = _unroll_reversible(f, z0, args=(W,), beta=beta, tol=tol, max_steps=max_steps)
    loss_u = (z1_u**2).sum()
    loss_u.backward()
    gradW_u = W.grad.detach().clone()

    # Custom reversible adjoint
    W.grad = None
    z1_r, steps_r, err_r = solve_reversible_adjoint(
        f, z0, args=(W,), params=[W], beta=beta, tol=tol, max_steps=max_steps
    )
    loss_r = (z1_r**2).sum()
    loss_r.backward()
    gradW_r = W.grad.detach().clone()

    assert steps_u == steps_r, f"steps mismatch: unroll={steps_u}, rev={steps_r}"
    _assert_allclose(torch.tensor(err_u), torch.tensor(err_r), rtol=1e-6, atol=1e-6, msg="error mismatch")
    _assert_allclose(z1_u.detach(), z1_r.detach(), rtol=1e-6, atol=1e-6, msg="z1 mismatch")
    _assert_allclose(gradW_u, gradW_r, rtol=1e-4, atol=1e-6, msg="gradW mismatch")


def test_revdeq_adjoint_matches_unroll_nonlinear():
    torch.manual_seed(0)
    device = "cpu"

    # Slightly richer map: f(z) = tanh(z @ W + b + u)
    W = torch.nn.Parameter(torch.randn(3, 3, device=device) * 0.1)
    b = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    u = torch.randn(1, 3, device=device) * 0.01  # constant input

    def f(z, args):
        W_, b_, u_ = args
        return torch.tanh(z @ W_ + b_ + u_)

    z0 = torch.randn(1, 3, device=device) * 0.01
    beta = 0.8
    tol = 1e-7
    max_steps = 6

    # Unrolled baseline
    W.grad = None
    b.grad = None
    z1_u, steps_u, err_u = _unroll_reversible(
        f, z0, args=(W, b, u), beta=beta, tol=tol, max_steps=max_steps
    )
    target = torch.zeros_like(z1_u)
    loss_u = torch.nn.functional.mse_loss(z1_u, target)
    loss_u.backward()
    gradW_u = W.grad.detach().clone()
    gradb_u = b.grad.detach().clone()

    # Custom reversible adjoint
    W.grad = None
    b.grad = None
    z1_r, steps_r, err_r = solve_reversible_adjoint(
        f, z0, args=(W, b, u), params=[W, b], beta=beta, tol=tol, max_steps=max_steps
    )
    loss_r = torch.nn.functional.mse_loss(z1_r, target)
    loss_r.backward()
    gradW_r = W.grad.detach().clone()
    gradb_r = b.grad.detach().clone()

    assert steps_u == steps_r, f"steps mismatch: unroll={steps_u}, rev={steps_r}"
    _assert_allclose(torch.tensor(err_u), torch.tensor(err_r), rtol=1e-6, atol=1e-6, msg="error mismatch")
    _assert_allclose(z1_u.detach(), z1_r.detach(), rtol=1e-6, atol=1e-6, msg="z1 mismatch")

    # Gradients should match closely (numerical differences expected but small)
    _assert_allclose(gradW_u, gradW_r, rtol=5e-3, atol=1e-6, msg="gradW mismatch")
    _assert_allclose(gradb_u, gradb_r, rtol=5e-3, atol=1e-6, msg="gradb mismatch")

