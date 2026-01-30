"""
PyTorch implementation of Reversible Deep Equilibrium Models (RevDEQ) for bilevel optimization.

This module ports the JAX-based reversible-deq implementation to PyTorch for use in
bilevel training of learned regularizers.

The key idea is to use a reversible fixed-point solver that allows efficient
backpropagation without storing intermediate states.

NUMERICAL STABILITY:
===================
The backward pass reconstructs states using: z0 = (z1 - beta * f(y1)) / (1 - beta)

This introduces error amplification of (1/(1-beta))^N over N steps:
  - beta=0.5: amplification = 2^N  (recommended, matches JAX default)
  - beta=0.8: amplification = 5^N  (unstable for N > ~10)
  - beta=0.9: amplification = 10^N (highly unstable)

Recommended parameters:
  - For verification (float64 NN): beta=0.5, steps<=15 → gradient error <1e-9
  - For training (float32 NN): beta=0.5, steps<=20 → gradient error <1e-6

When use_float64=True, all fixed-point operations (state updates, reconstructions)
are performed in float64 for numerical stability, matching JAX defaults. The neural
network evaluations remain in float32 for compatibility with pretrained weights.
"""

import torch
from typing import Callable, Tuple, Any, List, Optional
from dataclasses import dataclass

# Precision for fixed-point operations (matches JAX default)
FP_DTYPE = torch.float64
# Precision for neural network operations
NN_DTYPE = torch.float32


@dataclass
class Solution:
    """Solution from the reversible fixed-point solver."""
    z1: torch.Tensor
    steps: int
    error: float


def _call_function_fp64(function: Callable, z: torch.Tensor, args: Any, nn_dtype: Optional[torch.dtype] = NN_DTYPE) -> torch.Tensor:
    """
    Call the fixed-point function with proper dtype handling.
    
    - Convert input to nn_dtype (default: float32) for neural network evaluation
    - Convert output back to float64 for fixed-point math
    
    Args:
        function: The fixed-point function
        z: Input tensor (typically float64 for fixed-point precision)
        args: Additional arguments
        nn_dtype: The dtype expected by the neural network (default: float32)
    """
    if z.dtype != nn_dtype:
        z_nn = z.to(nn_dtype)
        result = function(z_nn, args)
        return result.to(FP_DTYPE)
    else:
        # Already in correct dtype - call directly
        result = function(z, args)
        return result.to(FP_DTYPE) if result.dtype != FP_DTYPE else result


class ReversibleSolver:
    """
    Reversible fixed-point solver.
    
    Solves z* = f(z*, args) using a reversible iteration scheme.
    The solver maintains state as (z, y) pairs and uses a beta parameter
    for the reversible updates.
    """
    
    def __init__(self, beta: float = 0.5, use_float64: bool = True, nn_dtype: Optional[torch.dtype] = None):
        """
        Args:
            beta: Relaxation parameter for reversible updates (0 < beta < 1).
                  Default 0.5 matches JAX implementation and provides good numerical stability.
                  Higher values (e.g., 0.8) can cause gradient errors to explode.
            use_float64: Use float64 for fixed-point operations
            nn_dtype: Dtype for neural network calls. If None, defaults to float32 when use_float64=True,
                      or no conversion when use_float64=False.
        """
        self.beta = beta
        self.use_float64 = use_float64
        # Default to float32 for NN calls when using float64 fixed-point (mixed precision)
        self.nn_dtype = nn_dtype if nn_dtype is not None else (NN_DTYPE if use_float64 else None)
    
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
        if self.use_float64:
            f0 = _call_function_fp64(function, z0, args, nn_dtype=self.nn_dtype)
        else:
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
        
        All fixed-point math is in float64 when use_float64=True.
        Neural network calls are in float32.
        
        Args:
            function: The fixed-point function f(z, args)
            z0: Current z value
            args: Additional arguments for the function
            solver_state: Current state (y0, f0)
            
        Returns:
            Tuple of (z1, new_state, error) where:
            - z1: Updated z value
            - new_state: Updated state (y1, f1)
            - error: Relative error estimate
        """
        y0, f0 = solver_state
        
        # Fixed-point update equations (all in float64 if use_float64)
        # y_{k+1} = (1 - beta) * y_k + beta * f(z_k)
        y1 = (1 - self.beta) * y0 + self.beta * f0
        
        # f(y_{k+1}) - call neural network
        if self.use_float64:
            f_y1 = _call_function_fp64(function, y1, args, nn_dtype=self.nn_dtype)
        else:
            f_y1 = function(y1, args)
        
        # z_{k+1} = (1 - beta) * z_k + beta * f(y_{k+1})
        z1 = (1 - self.beta) * z0 + self.beta * f_y1
        
        # f(z_{k+1}) - call neural network
        if self.use_float64:
            f_z1 = _call_function_fp64(function, z1, args, nn_dtype=self.nn_dtype)
        else:
            f_z1 = function(z1, args)

        # Compute error in float64 for accuracy
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
    - Uses float64 internally for numerical stability (matching JAX implementation)
    """

    @staticmethod
    def forward(ctx, z0: torch.Tensor, beta_t: torch.Tensor, tol_t: torch.Tensor, max_steps_t: torch.Tensor, *params):
        beta = float(beta_t.item())
        tol = float(tol_t.item())
        max_steps = int(max_steps_t.item())

        function = getattr(_ReversibleDEQFunction, "_py_function", None)
        args = getattr(_ReversibleDEQFunction, "_py_args", None)
        use_float64 = getattr(_ReversibleDEQFunction, "_use_float64", True)
        nn_dtype = getattr(_ReversibleDEQFunction, "_nn_dtype", NN_DTYPE)
        if function is None:
            raise RuntimeError("RevDEQ autograd: missing python `function` (internal wiring bug).")

        # Store original dtype for output conversion
        original_dtype = z0.dtype
        ctx.original_dtype = original_dtype
        ctx.nn_dtype = nn_dtype

        # Convert to float64 for fixed-point iterations if requested
        if use_float64:
            z0_fp = z0.to(FP_DTYPE)
        else:
            z0_fp = z0

        solver = ReversibleSolver(beta=beta, use_float64=use_float64, nn_dtype=nn_dtype)

        with torch.no_grad():
            y, fz = solver.init(function, z0_fp, args)
            z = z0_fp.clone()
            steps_taken = 0
            error = 2 * tol
            for i in range(max_steps):
                z, (y, fz), error = solver.step(function, z, args, (y, fz))
                steps_taken = i + 1
                if error < tol:
                    break

        # Save state in float64 for backward reconstruction
        ctx.save_for_backward(z.detach(), y.detach(), beta_t.detach())
        ctx.steps = steps_taken
        ctx.tol = tol
        ctx.max_steps = max_steps
        ctx.params_count = len(params)
        ctx.error = error
        ctx.params = list(params)
        ctx.args = args
        ctx.function = function
        ctx.use_float64 = use_float64

        # Convert output back to original dtype
        z_out = z.to(original_dtype)

        # Return z plus auxiliary tensors for logging/stats
        steps_out = torch.tensor(steps_taken, device=z0.device, dtype=torch.int64)
        err_out = torch.tensor(error, device=z0.device, dtype=original_dtype)
        return z_out, steps_out, err_out

    @staticmethod
    def backward(ctx, grad_z, grad_steps, grad_err, *unused):
        # Only grad_z matters (others are auxiliary outputs)
        z1, y1, beta_t = ctx.saved_tensors  # z1, y1 are in float64 if use_float64=True
        beta = float(beta_t.item())
        steps = int(getattr(ctx, "steps", 0))
        original_dtype = ctx.original_dtype
        use_float64 = ctx.use_float64

        function = ctx.function
        args = ctx.args
        params: List[torch.Tensor] = list(ctx.params)

        # Convert grad_z to float64 for numerical stability
        if use_float64:
            grad_z = grad_z.to(FP_DTYPE)

        # Initialize adjoint variables in float64
        grad_z1 = grad_z
        grad_y1 = torch.zeros_like(y1)

        # Initialize param gradients in float64 for accumulation accuracy
        if use_float64:
            grad_params = [torch.zeros(p.shape, dtype=FP_DTYPE, device=p.device) for p in params]
        else:
            grad_params = [torch.zeros_like(p) for p in params]

        # Use nn_dtype stored in context (set from solve_reversible_adjoint)
        nn_dtype = ctx.nn_dtype if hasattr(ctx, 'nn_dtype') else (params[0].dtype if params else original_dtype)
        
        # Reverse-time loop - all state reconstruction in float64
        for _ in range(steps):
            with torch.enable_grad():
                # Convert y1 to neural network dtype for function call
                y1_nn = y1.to(nn_dtype).detach().requires_grad_(True)
                fy1 = function(y1_nn, args)

                # VJP at y1: apply to grad_z1
                grads_y = torch.autograd.grad(
                    fy1,
                    (y1_nn, *params),
                    grad_outputs=grad_z1.to(nn_dtype),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                dgrad_y1 = grads_y[0] if grads_y[0] is not None else torch.zeros_like(y1_nn)
                dgrad_params_y = grads_y[1:]

                # Convert gradient to float64 for accumulation
                if use_float64:
                    dgrad_y1 = dgrad_y1.to(FP_DTYPE)

                # Gradient update (in float64)
                grad_y1 = grad_y1 + beta * dgrad_y1
                grad_y0 = (1 - beta) * grad_y1

                # State reconstruction in float64
                fy1_fp = fy1.detach().to(FP_DTYPE) if use_float64 else fy1.detach()
                z0 = (z1 - beta * fy1_fp) / (1 - beta)

                # Convert z0 to neural network dtype for function call
                z0_nn = z0.to(nn_dtype).detach().requires_grad_(True)
                fz0 = function(z0_nn, args)

                # VJP at z0: apply to grad_y1
                grads_z = torch.autograd.grad(
                    fz0,
                    (z0_nn, *params),
                    grad_outputs=grad_y1.to(nn_dtype),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                dgrad_z0 = grads_z[0] if grads_z[0] is not None else torch.zeros_like(z0_nn)
                dgrad_params_z = grads_z[1:]

                # Convert gradient to float64 for accumulation
                if use_float64:
                    dgrad_z0 = dgrad_z0.to(FP_DTYPE)

                # Gradient update (in float64)
                grad_z0 = (1 - beta) * grad_z1 + beta * dgrad_z0
                
                # Reconstruct y0 in float64
                fz0_fp = fz0.detach().to(FP_DTYPE) if use_float64 else fz0.detach()
                y0 = (y1 - beta * fz0_fp) / (1 - beta)

                # Accumulate parameter gradients in float64
                for i in range(len(params)):
                    gy = dgrad_params_y[i] if i < len(dgrad_params_y) and dgrad_params_y[i] is not None else 0
                    gz = dgrad_params_z[i] if i < len(dgrad_params_z) and dgrad_params_z[i] is not None else 0
                    if use_float64 and isinstance(gy, torch.Tensor):
                        gy = gy.to(FP_DTYPE)
                    if use_float64 and isinstance(gz, torch.Tensor):
                        gz = gz.to(FP_DTYPE)
                    grad_params[i] = grad_params[i] + beta * (gy + gz)

            # Step backwards (z0, y0 remain in float64)
            z1 = z0.detach()
            y1 = y0.detach()
            grad_z1 = grad_z0.detach()
            grad_y1 = grad_y0.detach()

        # Convert gradients back to original dtype
        if use_float64:
            grad_z1 = grad_z1.to(original_dtype)
            grad_params = [g.to(original_dtype) for g in grad_params]

        # Return grads for inputs: z0, beta_t, tol_t, max_steps_t, *params
        grad_beta = None
        grad_tol = None
        grad_max_steps = None
        return (grad_z1, grad_beta, grad_tol, grad_max_steps, *grad_params)


def solve_reversible(
    function: Callable,
    z0: torch.Tensor,
    args: Any,
    beta: float = 0.5,
    tol: float = 1e-3,
    max_steps: int = 50,
    use_float64: bool = True,
) -> Solution:
    """
    Solve the fixed-point equation z* = f(z*, args) using reversible iterations.
    
    Args:
        function: The fixed-point function f(z, args)
        z0: Initial guess
        args: Additional arguments for the function
        beta: Relaxation parameter (0 < beta < 1). Default 0.5 for numerical stability.
        tol: Convergence tolerance
        max_steps: Maximum number of iterations
        use_float64: Use float64 for fixed-point operations (default: True)
        
    Returns:
        Solution object containing z1, steps, and error
    """
    original_dtype = z0.dtype
    
    if use_float64:
        z0 = z0.to(FP_DTYPE)
    
    solver = ReversibleSolver(beta=beta, use_float64=use_float64)
    solver_state = solver.init(function, z0, args)
    z = z0.clone()
    
    steps_taken = 0
    for step in range(max_steps):
        z, solver_state, error = solver.step(function, z, args, solver_state)
        steps_taken = step + 1
        
        if error < tol:
            break
    
    # Convert back to original dtype
    z = z.to(original_dtype)
    
    return Solution(z1=z, steps=steps_taken, error=error)


def solve_reversible_adjoint(
    function: Callable,
    z0: torch.Tensor,
    args: Any,
    params: List[torch.Tensor],
    beta: float = 0.5,
    tol: float = 1e-3,
    max_steps: int = 50,
    use_float64: bool = True,
    nn_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, int, float]:
    """
    Reversible DEQ solve with a custom backward pass (RevDEQ-style adjoint).

    IMPORTANT: Use beta=0.5 (default) for numerical stability. Higher values (e.g., 0.8)
    can cause gradient errors to explode exponentially with the number of steps.

    Args:
        function: The fixed-point function f(z, args)
        z0: Initial guess
        args: Additional arguments for the function
        params: List of parameters to compute gradients for
        beta: Relaxation parameter (0 < beta < 1). Default 0.5 for numerical stability.
              Error amplification is (1/(1-beta))^N over N steps.
        tol: Convergence tolerance
        max_steps: Maximum number of iterations
        use_float64: Use float64 for fixed-point operations (default: True)
                     All state updates and reconstructions use float64.
        nn_dtype: Dtype for neural network calls. If None, defaults to float32 when use_float64=True
                  (mixed precision mode), or uses params dtype for float64 NNs in verification.

    Returns:
        z1: fixed point tensor (with custom gradient)
        steps: number of forward steps taken
        error: final residual
    """
    beta_t = torch.tensor(beta, device=z0.device, dtype=z0.dtype)
    tol_t = torch.tensor(tol, device=z0.device, dtype=z0.dtype)
    max_steps_t = torch.tensor(max_steps, device=z0.device, dtype=torch.int64)

    # Determine nn_dtype: use provided value, or default to float32 for mixed precision
    if nn_dtype is None:
        nn_dtype = NN_DTYPE if use_float64 else None

    # Attach python objects for forward/backward
    _ReversibleDEQFunction._py_function = function
    _ReversibleDEQFunction._py_args = args
    _ReversibleDEQFunction._use_float64 = use_float64
    _ReversibleDEQFunction._nn_dtype = nn_dtype

    z1, steps_t, err_t = _ReversibleDEQFunction.apply(z0, beta_t, tol_t, max_steps_t, *params)
    return z1, int(steps_t.item()), float(err_t.item())
