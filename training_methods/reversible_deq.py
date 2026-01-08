"""
PyTorch implementation of Reversible Deep Equilibrium Models (RevDEQ) for bilevel optimization.

This module ports the JAX-based reversible-deq implementation to PyTorch for use in
bilevel training of learned regularizers.

The key idea is to use a reversible fixed-point solver that allows efficient
backpropagation without storing intermediate states.
"""

import torch
from typing import Callable, Tuple, Any, List
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
        y0 = z0.clone()
        f0 = function(y0, args)
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
        
        # Reversible update: y1 = (1 - beta) * y0 + beta * f0
        y1 = (1 - self.beta) * y0 + self.beta * f0
        f_y1 = function(y1, args)  # f1 for the state
        
        # Reversible update: z1 = (1 - beta) * z0 + beta * f_y1
        z1 = (1 - self.beta) * z0 + self.beta * f_y1
        f_z1 = function(z1, args)  # For error computation only
        
        # Compute relative error using z1 and f_z1
        error = torch.norm(z1 - f_z1) / (1e-5 + torch.norm(f_z1))
        error = error.item()
        
        # Return state with (y1, f_y1) to maintain reversible scheme consistency
        return z1, (y1, f_y1), error


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
