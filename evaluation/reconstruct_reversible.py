"""
Reconstruction using reversible fixed-point solver for bilevel optimization.

This module provides a reconstruction function that uses the reversible DEQ
solver instead of nmAPG for solving the lower-level variational problem.
"""

import torch
from typing import Callable, Optional
from training_methods.reversible_deq import solve_reversible, ReversibleSolver


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
    beta=0.8,  # relaxation parameter for reversible solver (0 < beta < 1)
    verbose=False,  # set to True for some debug prints
    return_stats=False,  # return some statistics in addition to the reconstruction
):
    """
    Reconstruct image using reversible fixed-point solver.
    
    This solves the variational problem:
        min_x data_fidelity(x, y, physics) + lambda * regularizer(x)
    
    by formulating it as a fixed-point equation and solving with reversible iterations.
    
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
        beta: Relaxation parameter for reversible solver
        verbose: Print debug information
        return_stats: Return statistics dictionary
        
    Returns:
        Reconstructed image tensor (and stats if return_stats=True)
    """
    
    if x_init is not None:
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y)
    
    # Define the fixed-point function
    # The fixed-point equation is: x* = x* - step_size * grad(energy)
    # where energy = data_fidelity(x, y, physics) + lambda * regularizer(x)
    def fixed_point_function(z, args):
        """
        Fixed-point function for the variational problem.
        
        Args:
            z: Current iterate
            args: Tuple of (y, physics, data_fidelity, regularizer, lamda, step_size)
            
        Returns:
            Next iterate: z - step_size * grad(energy)
        """
        y_arg, physics_arg, data_fidelity_arg, regularizer_arg, lamda_arg, step_size_arg = args
        
        # Compute gradient of the energy
        grad_data = data_fidelity_arg.grad(z, y_arg, physics_arg)
        grad_reg = lamda_arg * regularizer_arg.grad(z)
        grad_total = grad_data + grad_reg
        
        # Fixed-point update: z_new = z - step_size * grad
        z_new = z - step_size_arg * grad_total
        
        return z_new
    
    # Prepare arguments for the fixed-point function
    args = (y, physics, data_fidelity, regularizer, lamda, step_size)
    
    # Solve using reversible iterations
    solution = solve_reversible(
        function=fixed_point_function,
        z0=x,
        args=args,
        beta=beta,
        tol=tol,
        max_steps=max_iter,
    )
    
    if verbose:
        print(f"Reversible solver converged in {solution.steps} steps with error {solution.error:.6e}")
    
    stats = dict(steps=solution.steps, error=solution.error, L=1.0/step_size)
    
    if return_stats:
        return solution.z1, stats
    return solution.z1
