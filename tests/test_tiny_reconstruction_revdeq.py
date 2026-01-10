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
from evaluation.reconstruct_reversible import reconstruct_reversible


class IdentityPhysics:
    """Minimal physics mock: y = x, and A_dagger(y) = y."""

    def __call__(self, x):
        return x

    def A_dagger(self, y):
        return y


class L2DataFidelity:
    """Minimal L2 data fidelity: 0.5||x - y||^2 and grad(x)=x-y."""

    def __call__(self, x, y, physics):
        return 0.5 * ((x - y) ** 2).view(x.shape[0], -1).sum(-1)

    def grad(self, x, y, physics):
        return x - y


class DummyReg(torch.nn.Module):
    """
    Simple trainable regularizer with grad(x)=w*x and energy g(x)=0.5*w*||x||^2.
    This is enough to validate the RevDEQ reconstruction path + gradients.
    """

    def __init__(self, w_init=0.1, dtype=torch.float64):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(float(w_init), dtype=dtype))

    def g(self, x):
        return 0.5 * self.w * (x**2).view(x.shape[0], -1).sum(-1)

    def grad(self, x, get_energy=False):
        if get_energy:
            return self.g(x), self.w * x
        return self.w * x


def test_tiny_reconstruction_revdeq_grad_matches_unroll():
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.float64

    physics = IdentityPhysics()
    data_fidelity = L2DataFidelity()
    regularizer = DummyReg(w_init=0.2, dtype=dtype).to(device)

    # Tiny "image": 1x1x2x2
    x_gt = torch.tensor([[[[0.1, -0.2], [0.05, 0.3]]]], dtype=dtype, device=device)
    y = physics(x_gt)

    lam = 1.0
    step_size = 0.5
    beta = 0.8
    tol = 1e-12
    max_iter = 6

    # --- Reference: unrolled reversible iterations with standard autograd ---
    regularizer.w.grad = None
    x0 = physics.A_dagger(y).detach()

    def fixed_point(z, args):
        y_arg, physics_arg, data_fid_arg, reg_arg, lam_arg, step_arg = args
        grad_data = data_fid_arg.grad(z, y_arg, physics_arg)
        grad_reg = lam_arg * reg_arg.grad(z)
        return z - step_arg * (grad_data + grad_reg)

    args = (y, physics, data_fidelity, regularizer, lam, step_size)
    solver = ReversibleSolver(beta=beta)
    yy, fz = solver.init(fixed_point, x0, args)
    z = x0.clone()
    for _ in range(max_iter):
        z, (yy, fz), _err = solver.step(fixed_point, z, args, (yy, fz))

    loss_u = ((z - x_gt) ** 2).mean()
    loss_u.backward()
    grad_w_unroll = regularizer.w.grad.detach().clone()

    # --- RevDEQ path used in code: reconstruct_reversible + custom backward ---
    regularizer.w.grad = None
    x_rec = reconstruct_reversible(
        y,
        physics,
        data_fidelity,
        regularizer,
        lam,
        step_size,
        max_iter,
        tol,
        x_init=None,
        beta=beta,
        verbose=False,
        return_stats=False,
    )
    loss_r = ((x_rec - x_gt) ** 2).mean()
    loss_r.backward()
    grad_w_rev = regularizer.w.grad.detach().clone()

    torch.testing.assert_close(grad_w_unroll, grad_w_rev, rtol=5e-3, atol=1e-10)

