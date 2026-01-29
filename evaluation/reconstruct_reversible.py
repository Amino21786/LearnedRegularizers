"""
Reconstruction using reversible fixed-point solver for bilevel optimization.

This module provides a reconstruction function that uses the reversible DEQ
solver instead of nmAPG for solving the lower-level variational problem.

The reversible iteration scheme (from RevDEQ) solves z* = f(z*) via:
    y_{k+1} = (1 - β) * y_k + β * f(y_k)
    z_{k+1} = (1 - β) * z_k + β * f(y_{k+1})

where f(z) = z - step_size * ∇(energy) is the gradient descent operator.
This can be reversed for memory-efficient backpropagation.
"""

import torch
from typing import Optional


def reconstruct_reversible(
    y,  # observation in the variational problem
    physics,  # deepinv physics object defining the forward operator and the noise model
    data_fidelity,  # deepinv data fidelity object defining the data fidelity term
    regularizer,  # regularizer in the variational problem
    lamda,  # regularization parameter
    step_size,  # step size for the fixed-point iteration
    max_iter,  # maximal number of iterations
    tol,  # tolerance for the stopping criterion (relative residual)
    x_init=None,  # initialization (None for using physics.A_dagger(y))
    beta=0.8,  # relaxation parameter for reversible solver (0 < beta <= 1)
    verbose=False,  # set to True for some debug prints
    return_stats=False,  # return some statistics in addition to the reconstruction
    use_embedded_beta=False,  # if True, embed beta directly in fixed_point_function (experimental)
    use_float64=True,  # use float64 for internal computations (matches JAX, improves gradient accuracy)
):
    """
    Reconstruct image using reversible fixed-point solver.
    
    This solves the variational problem:
        min_x data_fidelity(x, y, physics) + lambda * regularizer(x)
    
    by formulating it as a fixed-point equation and solving with reversible iterations.
    
    The reversible scheme finds z* where z* = f(z*), with f being the gradient descent
    operator. The beta parameter controls the relaxation:
    
    Standard mode (use_embedded_beta=False):
        - f(z) = z - step_size * grad(energy)
        - Solver applies: z_{k+1} = (1-β)*z_k + β*f(y_{k+1})
        
    Embedded beta mode (use_embedded_beta=True, experimental):
        - f(z) directly incorporates the relaxation scheme
        - This matches the RevDEQ paper formulation more explicitly
    
    Args:
        y: Observation tensor
        physics: Physics operator (deepinv physics object)
        data_fidelity: Data fidelity term
        regularizer: Regularizer function
        lamda: Regularization parameter
        step_size: Step size for fixed-point iteration
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        x_init: Initial guess (None for physics.A_dagger(y))
        beta: Relaxation parameter for reversible solver (0 < beta <= 1)
              beta < 1: underrelaxation (more stable)
              beta = 1: standard fixed-point iteration (Anderson acceleration style)
              beta > 1: overrelaxation (may diverge)
        verbose: Print debug information
        return_stats: Return statistics dictionary
        use_embedded_beta: Embed beta in fixed_point_function (experimental)
        use_float64: Use float64 for internal reversible computations (default: True)
                     This matches the JAX reference implementation and significantly
                     improves gradient accuracy by reducing reconstruction errors.
        
    Returns:
        Reconstructed image tensor (and stats if return_stats=True)
    """
    
    # Lazy import to avoid circular dependency with training_methods
    from training_methods.reversible_deq import solve_reversible_adjoint
    
    if x_init is not None:
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y)
    
    # =========================================================================
    # FIXED-POINT FUNCTION DEFINITION
    # =========================================================================
    # The variational problem is: min_x E(x) = D(x,y) + λ*R(x)
    # where D is data fidelity and R is the regularizer.
    #
    # We solve this by finding the fixed point of the gradient descent operator:
    #     T(z) = z - step_size * ∇E(z)
    #
    # At the fixed point z*: z* = T(z*), which means ∇E(z*) = 0.
    #
    # The reversible solver wraps this with the beta-averaged iteration scheme.
    # =========================================================================
    
    if use_embedded_beta:
        # EXPERIMENTAL: Embed beta directly into the fixed-point function
        # This formulation explicitly shows the RevDEQ relaxation scheme.
        # 
        # WARNING: This changes the mathematical formulation - use with caution!
        # The solver will apply beta AGAIN, so this effectively squares the effect.
        # 
        # Original (keep for reference):
        # def fixed_point_function(z, args):
        #     y_arg, physics_arg, data_fidelity_arg, regularizer_arg, lamda_arg, step_size_arg, _ = args
        #     grad_data = data_fidelity_arg.grad(z, y_arg, physics_arg)
        #     grad_reg = lamda_arg * regularizer_arg.grad(z)
        #     grad_total = grad_data + grad_reg
        #     z_new = z - step_size_arg * grad_total
        #     return z_new
        
        def fixed_point_function(z, args):
            """
            Fixed-point function with embedded beta relaxation (EXPERIMENTAL).
            
            This directly incorporates the RevDEQ relaxation into the operator:
                f(z) = (1 - β) * z + β * (z - step_size * ∇E(z))
                     = z - β * step_size * ∇E(z)
            
            Note: The solver will NOT apply beta again when use_embedded_beta=True,
            effectively making beta=1 in the solver.
            
            Args:
                z: Current iterate
                args: Tuple containing problem parameters
                
            Returns:
                Relaxed gradient descent update
            """
            y_arg, physics_arg, data_fidelity_arg, regularizer_arg, lamda_arg, step_size_arg, beta_arg = args
            
            # Compute gradient of the energy E(z) = D(z,y) + λ*R(z)
            grad_data = data_fidelity_arg.grad(z, y_arg, physics_arg)
            grad_reg = lamda_arg * regularizer_arg.grad(z)
            grad_total = grad_data + grad_reg
            
            # Embedded beta relaxation: z_new = z - β * step_size * grad
            # This is equivalent to: z_new = (1-β)*z + β*(z - step_size*grad)
            z_new = z - beta_arg * step_size_arg * grad_total
            
            return z_new
        
        # Include beta in args, solver will use beta=1.0
        args = (y, physics, data_fidelity, regularizer, lamda, step_size, beta)
        solver_beta = 1.0  # Don't apply beta again in solver
        
    else:
        # STANDARD: Beta is applied by the ReversibleSolver, not here
        # This is the recommended approach and matches the RevDEQ architecture.
        #
        # The solver performs:
        #     y_{k+1} = (1 - β) * y_k + β * f(y_k)
        #     z_{k+1} = (1 - β) * z_k + β * f(y_{k+1})
        #
        # where f(z) = z - step_size * ∇E(z) is defined below.
        
        def fixed_point_function(z, args):
            """
            Standard gradient descent fixed-point function.
            
            The fixed point of T(z) = z - step_size * ∇E(z) is the minimizer of E(z).
            The reversible solver wraps this with beta-averaged iterations.
            
            Args:
                z: Current iterate
                args: Tuple of (y, physics, data_fidelity, regularizer, lamda, step_size)
                
            Returns:
                Gradient descent update: z - step_size * grad(energy)
            """
            y_arg, physics_arg, data_fidelity_arg, regularizer_arg, lamda_arg, step_size_arg = args
            
            # Compute gradient of the energy E(z) = D(z,y) + λ*R(z)
            grad_data = data_fidelity_arg.grad(z, y_arg, physics_arg)
            grad_reg = lamda_arg * regularizer_arg.grad(z)
            grad_total = grad_data + grad_reg
            
            # Standard gradient descent step
            z_new = z - step_size_arg * grad_total
            
            return z_new
        
        args = (y, physics, data_fidelity, regularizer, lamda, step_size)
        solver_beta = beta  # Let solver apply beta

    # =========================================================================
    # SOLVE USING REVERSIBLE ITERATIONS
    # =========================================================================
    # The custom autograd function handles:
    # - Forward: reversible fixed-point iterations (memory efficient)
    # - Backward: reconstruct states in reverse, accumulate VJPs for parameters
    # =========================================================================
    
    params = [p for p in regularizer.parameters() if p.requires_grad]
    z1, steps_taken, error = solve_reversible_adjoint(
        function=fixed_point_function,
        z0=x,
        args=args,
        params=params,
        beta=solver_beta,
        tol=tol,
        max_steps=max_iter,
        use_float64=use_float64,
    )

    if verbose:
        print(
            f"Reversible solver converged in {steps_taken} steps with error {error:.6e}"
        )

    stats = dict(steps=steps_taken, error=error, L=torch.tensor(1.0 / step_size))
    
    if return_stats:
        return z1, stats
    return z1
