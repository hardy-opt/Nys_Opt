# Theoretical Background

## Introduction

The Nys-Newton optimizer implements a novel second-order optimization method that efficiently approximates curvature information using the Nyström method. This approach addresses the computational challenges of traditional second-order methods while maintaining their superior convergence properties.

## Mathematical Foundation

### Problem Formulation

Consider the optimization problem:
min f(x)
where f: ℝⁿ → ℝ is a twice-differentiable function.

### Newton's Method

The classical Newton's method uses the update rule:
x_{k+1} = x_k - α H_k^{-1} ∇f(x_k)
where:
- H_k is the Hessian matrix at iteration k
- ∇f(x_k) is the gradient
- α is the step size

**Challenges:**
1. Computing H_k^{-1} is O(n³) for n parameters
2. Storing the full Hessian requires O(n²) memory
3. The Hessian may not be positive definite

### Nyström Approximation

The Nyström method approximates a positive semi-definite matrix M ∈ ℝⁿˣⁿ using a low-rank approximation:
M ≈ M̃ = C W^† C^T

where:
- C ∈ ℝⁿˣᵐ is formed by m randomly selected columns of M
- W ∈ ℝᵐˣᵐ is the intersection of selected rows and columns
- W^† is the Moore-Penrose pseudoinverse of W
- m << n (typically m = O(log n))

### Nys-Newton Algorithm

The Nys-Newton method applies Nyström approximation to the Hessian:

1. **Sample columns**: Randomly select m columns of the Hessian
2. **Form approximation**: Compute H̃_k ≈ C W^† C^T
3. **Update**: x_{k+1} = x_k - α (H̃_k + λI)^{-1} ∇f(x_k)

where λ > 0 is a damping parameter for numerical stability.

## Computational Complexity

| Method | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| Newton | O(n³) | O(n²) |
| Nys-Newton | O(mn² + m³) | O(mn) |
| First-order | O(n) | O(n) |

For m = O(log n), Nys-Newton achieves near-linear complexity while maintaining second-order convergence properties.

## Convergence Analysis

### Local Convergence

**Theorem**: Under standard assumptions (strong convexity, Lipschitz continuity of Hessian), the Nys-Newton method achieves superlinear convergence:
||x_{k+1} - x*|| ≤ C ||x_k - x*||^{1+ρ}

where ρ > 0 depends on the approximation quality and C is a constant.

### Global Convergence

With appropriate step size selection (e.g., backtracking line search), global convergence to statically optimal points is guaranteed for convex functions.

## Practical Considerations

### Rank Selection

The approximation rank m should balance accuracy and efficiency:
- **Too small**: Poor approximation, slow convergence
- **Too large**: High computational cost
- **Rule of thumb**: m = min(50, n/10) often works well

### Update Frequency

The Hessian approximation need not be updated every iteration:
- Update every k iterations (k = 5-20 typical)
- Reduces computational overhead
- Maintains good convergence properties

### Damping Strategy

The damping parameter λ ensures positive definiteness:
- **Adaptive damping**: Adjust based on gradient norm
- **Fixed damping**: Use small constant (1e-6 to 1e-4)
- **Levenberg-Marquardt**: λ = max(λ_min, ||∇f||)

## Comparison with Other Methods

### vs. L-BFGS

| Aspect | Nys-Newton | L-BFGS |
|--------|------------|--------|
| Curvature Info | Explicit Hessian approx | Implicit via gradients |
| Memory | O(mn) | O(kn) where k is history |
| Convergence | Superlinear | Superlinear |
| Parallelization | High | Limited |

### vs. Adam/SGD

| Aspect | Nys-Newton | Adam | SGD |
|--------|------------|------|-----|
| Order | Second | First | First |
| Convergence Rate | Superlinear | Sublinear | Linear |
| Per-iteration Cost | O(mn²) | O(n) | O(n) |
| Robustness | High | Medium | Low |

## Implementation Notes

### Numerical Stability

1. **Eigenvalue clipping**: Ensure eigenvalues > ε
2. **Condition number monitoring**: Track matrix conditioning
3. **Fallback strategies**: Switch to first-order if needed

### Computational Optimizations

1. **Matrix-free operations**: Avoid explicit Hessian construction
2. **Iterative solvers**: Use CG for linear systems
3. **Warm starts**: Reuse previous approximations

## References

1. Original paper: "Nyström-Approximated Curvature for Stochastic Optimization"
2. Nyström method: "Using the Nyström Method to Speed Up Kernel Machines"
3. Second-order optimization: "Numerical Optimization" by Nocedal & Wright