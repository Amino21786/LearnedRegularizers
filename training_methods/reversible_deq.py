"""
PyTorch implementation of Reversible Deep Equilibrium Models (RevDEQ) for bilevel optimization.

This module ports the JAX-based reversible-deq implementation to PyTorch for use in
bilevel training of learned regularizers.

The key idea is to use a reversible fixed-point solver that allows efficient
backpropagation without storing intermediate states.
"""

import torch
from typing import Callable, Tuple, Any, List, Optional
from dataclasses import dataclass


@dataclass
class Solution:
    """Solution from the reversible fixed-point solver."""
    z1: torch.Tensor
    steps: int
    error: float


class ReversibleSolver:
    """
    Reversible fixed-point solver.
    
    Solves z* = f(z*, args) using a reversible iteration scheme.
    The solver maintains state as (z, y) pairs and uses a beta parameter
    for the reversible updates.
    """
    
    def __init__(self, beta: float = 0.8):
        """
        Args:
            beta: Relaxation parameter for reversible updates (0 < beta < 1)
        """
        self.beta = beta
    
    def init(self, function: Callable, z0: torch.Tensor, args: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize solver state.
        
        Args:
            function: The fixed-point function f(z, args)
            z0: Initial guess
            args: Additional arguments for the function
            
        Returns:
            Tuple of (y0, f0) where y0 = z0 and f0 = f(z0, args)
        """
        # Match reversible-deq JAX implementation:
        # state is (y0, f(z0)) with y0 initialized to z0.
        y0 = z0.clone()
        f0 = function(z0, args)
        return y0, f0
    
    def step(
        self,
        function: Callable,
        z0: torch.Tensor,
        args: Any,
        solver_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], float]:
        """
        Perform one step of the reversible fixed-point iteration.
        
        Args:
            function: The fixed-point function f(z, args)
            z0: Current z value
            args: Additional arguments for the function
            solver_state: Current state (y0, f0)
            
        Returns:
            Tuple of (z1, new_state, error) where:
            - z1: Updated z value
            - new_state: Updated state (y1, f1) where f1 = function(y1, args)
            - error: Relative error estimate
        """
        y0, f0 = solver_state
        
        # Match reversible-deq JAX implementation exactly:
        # y_{k+1} uses f(z_k) (stored as f0), z_{k+1} uses f(y_{k+1}),
        # and the stored state keeps f(z_{k+1}) for the next step.
        y1 = (1 - self.beta) * y0 + self.beta * f0
        f_y1 = function(y1, args)
        z1 = (1 - self.beta) * z0 + self.beta * f_y1
        f_z1 = function(z1, args)

        error = torch.norm(z1 - f_z1) / (1e-5 + torch.norm(f_z1))
        error = float(error.item())

        return z1, (y1, f_z1), error


class _ReversibleDEQFunction(torch.autograd.Function):
    """
    Memory-efficient reversible DEQ solve with a custom backward pass.

    This is a PyTorch port of the JAX reversible-deq solver+adjoint:
    - Forward: runs reversible fixed-point iterations without building an autograd graph
    - Backward: reconstructs states in reverse and accumulates vjps w.r.t. parameters

    Notes:
    - Supports tensor state z only (sufficient for LearnedRegularizers reconstruction)
    - Exposes gradients for z0 and for any parameters explicitly passed as inputs
    """

    @staticmethod
    def forward(ctx, z0: torch.Tensor, beta_t: torch.Tensor, tol_t: torch.Tensor, max_steps_t: torch.Tensor, *params):
        beta = float(beta_t.item())
        tol = float(tol_t.item())
        max_steps = int(max_steps_t.item())

        function = getattr(_ReversibleDEQFunction, "_py_function", None)
        args = getattr(_ReversibleDEQFunction, "_py_args", None)
        if function is None:
            raise RuntimeError("RevDEQ autograd: missing python `function` (internal wiring bug).")

        solver = ReversibleSolver(beta=beta)

        with torch.no_grad():
            y, fz = solver.init(function, z0, args)
            z = z0.clone()
            steps_taken = 0
            error = 2 * tol
            for i in range(max_steps):
                z, (y, fz), error = solver.step(function, z, args, (y, fz))
                steps_taken = i + 1
                if error < tol:
                    break

        # Save minimal residual state for backward
        ctx.save_for_backward(z.detach(), y.detach(), beta_t.detach())
        ctx.steps = steps_taken
        ctx.tol = tol
        ctx.max_steps = max_steps
        ctx.params_count = len(params)
        ctx.error = error
        ctx.params = list(params)
        ctx.args = args
        ctx.function = function

        # Return z plus auxiliary tensors for logging/stats (no grads needed for these)
        steps_out = torch.tensor(steps_taken, device=z0.device, dtype=torch.int64)
        err_out = torch.tensor(error, device=z0.device, dtype=z0.dtype)
        return z, steps_out, err_out

    @staticmethod
    def backward(ctx, grad_z, grad_steps, grad_err, *unused):
        # Only grad_z matters (others are auxiliary outputs)
        z1, y1, beta_t = ctx.saved_tensors
        beta = float(beta_t.item())
        steps = int(getattr(ctx, "steps", 0))

        function = ctx.function
        args = ctx.args

        # Params are appended after the first 4 inputs; retrieve them from ctx by closure:
        # In PyTorch custom autograd, we don't get params directly here; but we can access
        # them as "inputs" only via the wrapper's captured list by storing in ctx.
        params: List[torch.Tensor] = list(ctx.params)

        # Initialize adjoint variables
        grad_z1 = grad_z
        grad_y1 = torch.zeros_like(y1)

        grad_params = [torch.zeros_like(p) for p in params]

        # Reverse-time loop
        for _ in range(steps):
            # Compute z0 from z1 and f(y1)
            with torch.enable_grad():
                y1_req = y1.detach().requires_grad_(True)
                fy1 = function(y1_req, args)

                # VJP at y1: apply to grad_z1
                grads_y = torch.autograd.grad(
                    fy1,
                    (y1_req, *params),
                    grad_outputs=grad_z1,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                dgrad_y1 = grads_y[0] if grads_y[0] is not None else torch.zeros_like(y1_req)
                dgrad_params_y = grads_y[1:]

                grad_y1 = grad_y1 + beta * dgrad_y1
                grad_y0 = (1 - beta) * grad_y1

                z0 = (z1 - beta * fy1.detach()) / (1 - beta)

                # Compute y0 from y1 and f(z0)
                z0_req = z0.detach().requires_grad_(True)
                fz0 = function(z0_req, args)

                # VJP at z0: apply to grad_y1
                grads_z = torch.autograd.grad(
                    fz0,
                    (z0_req, *params),
                    grad_outputs=grad_y1,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                dgrad_z0 = grads_z[0] if grads_z[0] is not None else torch.zeros_like(z0_req)
                dgrad_params_z = grads_z[1:]

                grad_z0 = (1 - beta) * grad_z1 + beta * dgrad_z0
                y0 = (y1 - beta * fz0.detach()) / (1 - beta)

                # Accumulate parameter gradients (beta * (vjp_y + vjp_z))
                for i in range(len(params)):
                    gy = dgrad_params_y[i] if i < len(dgrad_params_y) and dgrad_params_y[i] is not None else 0
                    gz = dgrad_params_z[i] if i < len(dgrad_params_z) and dgrad_params_z[i] is not None else 0
                    grad_params[i] = grad_params[i] + beta * (gy + gz)

            # Step backwards
            z1 = z0.detach()
            y1 = y0.detach()
            grad_z1 = grad_z0.detach()
            grad_y1 = grad_y0.detach()

        # Return grads for inputs: z0, beta_t, tol_t, max_steps_t, *params
        grad_beta = None
        grad_tol = None
        grad_max_steps = None
        return (grad_z1, grad_beta, grad_tol, grad_max_steps, *grad_params)


def solve_reversible(
    function: Callable,
    z0: torch.Tensor,
    args: Any,
    beta: float = 0.8,
    tol: float = 1e-3,
    max_steps: int = 50,
) -> Solution:
    """
    Solve the fixed-point equation z* = f(z*, args) using reversible iterations.
    
    Args:
        function: The fixed-point function f(z, args)
        z0: Initial guess
        args: Additional arguments for the function
        beta: Relaxation parameter (0 < beta < 1)
        tol: Convergence tolerance
        max_steps: Maximum number of iterations
        
    Returns:
        Solution object containing z1, steps, and error
    """
    solver = ReversibleSolver(beta=beta)
    solver_state = solver.init(function, z0, args)
    z = z0.clone()
    
    steps_taken = 0
    for step in range(max_steps):
        z, solver_state, error = solver.step(function, z, args, solver_state)
        steps_taken = step + 1
        
        if error < tol:
            break
    
    return Solution(z1=z, steps=steps_taken, error=error)


def solve_reversible_adjoint(
    function: Callable,
    z0: torch.Tensor,
    args: Any,
    params: List[torch.Tensor],
    beta: float = 0.8,
    tol: float = 1e-3,
    max_steps: int = 50,
) -> Tuple[torch.Tensor, int, float]:
    """
    Reversible DEQ solve with a custom backward pass (RevDEQ-style adjoint).

    Returns:
        z1: fixed point tensor (with custom gradient)
        steps: number of forward steps taken
        error: final residual
    """
    beta_t = torch.tensor(beta, device=z0.device, dtype=z0.dtype)
    tol_t = torch.tensor(tol, device=z0.device, dtype=z0.dtype)
    max_steps_t = torch.tensor(max_steps, device=z0.device, dtype=torch.int64)

    # Attach python objects for forward/backward (non-tensors cannot be passed through apply)
    _ReversibleDEQFunction._py_function = function
    _ReversibleDEQFunction._py_args = args

    z1, steps_t, err_t = _ReversibleDEQFunction.apply(z0, beta_t, tol_t, max_steps_t, *params)
    return z1, int(steps_t.item()), float(err_t.item())
