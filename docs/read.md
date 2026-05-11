---
title: Optimization Techniques and Their Relevance to Neural Networks
author: Generated from optimization_code.py
documentclass: article
---

This document explains the ideas contained in the file `optimization_code.py`, focusing on their significance for neural network training. Every variable appearing in equations is defined in detail. The code implements classical unconstrained and constrained optimization methods that underpin modern deep learning.

# Unconstrained Optimisation Methods

Unconstrained optimisation forms the backbone of learning. The objective is to find \(\theta^* = \arg\min_\theta f(\theta)\) without any restrictions on \(\theta\).

## Gradient Descent (fixed learning rate)

Update rule:

\[
\theta_{k+1} = \theta_k - \alpha\,\nabla f(\theta_k)
\]

### Variable definitions

- \(\theta_k \in \mathbb{R}^n\) : the parameter vector at iteration \(k\). In a neural network it represents all weights and biases flattened into a single vector.
- \(\alpha \in \mathbb{R}^+\) : the learning rate (step size), a fixed positive scalar. In deep learning typical values range from \(10^{-5}\) to \(1.0\).
- \(\nabla f(\theta_k) \in \mathbb{R}^n\) : the gradient of the scalar loss function \(f\) with respect to \(\theta\), evaluated at \(\theta_k\). For neural networks this is computed by back‑propagation.
- \(k\) : iteration index (discrete time step).
- \(f\) : the objective (loss) function, e.g., mean squared error or cross‑entropy.

The term \(\alpha\,\nabla f(\theta_k)\) is a scaled correction vector; subtracting it moves the parameters along the steepest descent direction, scaled by \(\alpha\).

## Steepest Descent (Armijo backtracking line search)

The Armijo condition used in the code:

\[
f(\theta_k + \alpha d_k) \le f(\theta_k) + c\,\alpha\,\nabla f(\theta_k)^{\mathsf{T}} d_k
\]

with \(d_k = -\nabla f(\theta_k)\).

### Variable definitions

- \(\theta_k\) : current parameter vector.
- \(d_k \in \mathbb{R}^n\) : descent direction; here simply the negative gradient \(-\nabla f(\theta_k)\).
- \(\alpha\) : trial step length, initially set to \(1.0\) and reduced by factor \(\rho\) until the condition holds.
- \(f(\theta_k)\) : current objective value (scalar).
- \(f(\theta_k + \alpha d_k)\) : objective value at the candidate point.
- \(c \in (0,1)\) : Armijo constant, often \(10^{-4}\); it controls the sufficient decrease requirement. A smaller value makes the condition more lenient.
- \(\nabla f(\theta_k)^{\mathsf{T}} d_k\) : directional derivative along \(d_k\); it is negative if \(d_k\) is a descent direction. Here it equals \(-\|\nabla f(\theta_k)\|^2\).
- \(\rho \in (0,1)\) : backtracking factor (e.g., \(0.5\)). Each trial multiplies \(\alpha\) by \(\rho\).
- The product \(c\,\alpha\,\nabla f^{\mathsf{T}} d_k\) is the minimum required linear reduction.

In deep learning, complete line searches are rarely used, but the concept of adaptive step sizes (like those in Adam) echoes the idea of tuning the effective learning rate per iteration.

## Newton’s Method

Update rule:

\[
\theta_{k+1} = \theta_k - H^{-1}\,\nabla f(\theta_k)
\]

### Variable definitions

- \(\theta_k, \nabla f(\theta_k)\) as before.
- \(H \in \mathbb{R}^{n \times n}\) : the Hessian matrix of \(f\) at \(\theta_k\), i.e., the matrix of second partial derivatives \(H_{ij} = \frac{\partial^2 f}{\partial \theta_i \partial \theta_j}\). For a twice‑differentiable neural network loss this matrix can be very large.
- \(H^{-1}\) : the inverse of the Hessian. Multiplying the gradient by \(H^{-1}\) rescales the step according to local curvature, yielding faster convergence on ill‑conditioned surfaces.
- The direction \(d = -H^{-1}\nabla f\) is called the Newton direction. For quadratic functions it points directly to the minimum.

Because computing and inverting the full Hessian is \(O(n^3)\) and memory‑intensive, direct Newton is impractical for large networks. It inspires approximate second‑order methods that exploit curvature in a more efficient manner.

## BFGS Quasi‑Newton Method

The method maintains an approximation \(B_k\) (or often \(H_k\), the approximate inverse Hessian) that satisfies the secant equation \(B_{k+1}s_k = y_k\). The code updates the inverse Hessian approximation \(H\) with:

\[
\begin{aligned}
s_k &= \theta_{k+1} - \theta_k \quad \text{(parameter step)}\\
y_k &= \nabla f(\theta_{k+1}) - \nabla f(\theta_k) \quad \text{(gradient change)}\\
\rho_k &= \frac{1}{y_k^{\mathsf{T}} s_k}\\
H_{k+1} &= (I - \rho_k s_k y_k^{\mathsf{T}})\,H_k\,(I - \rho_k y_k s_k^{\mathsf{T}}) + \rho_k s_k s_k^{\mathsf{T}}
\end{aligned}
\]

### Variable definitions

- \(s_k \in \mathbb{R}^n\) : the difference between successive parameter vectors. It indicates the direction and magnitude of the step taken.
- \(y_k \in \mathbb{R}^n\) : the change in gradients. It captures how the gradient field varied along \(s_k\), approximating the action of the Hessian.
- \(\rho_k\) : a scalar (\(1 / (y_k^{\mathsf{T}} s_k)\)). The inner product \(y_k^{\mathsf{T}} s_k\) must be positive for the approximate Hessian to remain positive definite; the code guards against near‑zero values.
- \(H_k \in \mathbb{R}^{n \times n}\) : the current approximate inverse Hessian. It is initialised as the identity matrix.
- \(I \in \mathbb{R}^{n \times n}\) : identity matrix.
- The outer products \(s_k y_k^{\mathsf{T}}\) and \(y_k s_k^{\mathsf{T}}\) form rank‑1 matrices; together they produce a rank‑2 update that satisfies the secant equation while preserving symmetry and positive definiteness.
- The resulting direction is \(d_k = -H_{k+1}\nabla f(\theta_k)\).

In neural networks, the limited‑memory variant L‑BFGS stores only a few previous \((s, y)\) pairs and applies the update without forming the full matrix, making it usable for moderately sized problems.

### Comparison of Unconstrained Methods

| Method           | Update Rule                                                | Pros                                                       | Cons                                                            | Relevance to Neural Networks                                               |
|------------------|------------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------------------|
| Gradient Descent | \(\theta_{k+1} = \theta_k - \alpha\nabla f\)               | Simple, cheap per iteration                                | Sensitive to \(\alpha\); slow on ill‑conditioned problems       | Foundation of SGD; used in all minibatch training                          |
| Steepest Descent | Direction \(-\nabla f\), step chosen by Armijo line search | Robust step size; guaranteed decrease                      | Requires function evaluations; expensive for large networks     | Inspires learning‑rate schedules and warmup strategies                     |
| Newton’s Method  | \(\theta_{k+1} = \theta_k - H^{-1}\nabla f\)               | Quadratic convergence near optimum; handles curvature well | \(O(n^3)\) Hessian inversion; \(H\) may be indefinite           | Motivates approximate second‑order methods (K‑FAC, Shampoo)                |
| BFGS             | Inverse Hessian updated via rank‑two secant condition      | Superlinear convergence; no Hessian computation            | Memory \(O(n^2)\) for full matrix; limited to mid‑size problems | L‑BFGS used for small‑scale fine‑tuning; benchmark for optimizer behaviour |

# Constrained Optimisation

## Lagrange Multipliers (Equality Constraints)

Problem: \(\min_{x,y} \;(x^2+y^2) \;\; \text{s.t.} \;\; x+y-4 = 0\).

The Lagrangian is \(\mathcal{L}(x,y,\lambda) = x^2 + y^2 + \lambda(x+y-4)\).

Stationarity conditions:

\[
\begin{cases}
\frac{\partial\mathcal{L}}{\partial x} = 2x + \lambda = 0\\
\frac{\partial\mathcal{L}}{\partial y} = 2y + \lambda = 0\\
\frac{\partial\mathcal{L}}{\partial\lambda} = x+y-4 = 0
\end{cases}
\]

### Variable definitions

- \(x, y \in \mathbb{R}\) : optimisation variables.
- \(\lambda \in \mathbb{R}\) : Lagrange multiplier. It measures the sensitivity of the optimal objective to changes in the constraint. At the solution, \(\lambda = 4\).
- \(\nabla f = (2x, 2y)\) : gradient of the objective.
- \(\nabla g = (1, 1)\) : gradient of the constraint function \(g(x,y)=x+y-4\).
- The stationarity condition expresses that \(\nabla f = -\lambda \nabla g\); i.e., the two gradients are parallel. This means at the optimum we cannot move along the constraint and reduce the objective further.

## KKT Conditions (Inequality Constraints)

Problem: \(\min_{x,y} \;(x^2+y^2) \;\; \text{s.t.} \;\; x+y-4 \ge 0\).

The KKT conditions at a local minimum \(x^*\) are:

1. **Stationarity**: \(\nabla f(x^*) = \mu \nabla g(x^*)\)
2. **Primal feasibility**: \(g(x^*) \ge 0\)
3. **Dual feasibility**: \(\mu \ge 0\)
4. **Complementary slackness**: \(\mu \, g(x^*) = 0\)

### Variable definitions

- \(x^* = (x^*, y^*)\) : optimal point, found to be \((2,2)\).
- \(\mu \in \mathbb{R}\) : KKT multiplier for the inequality constraint. It plays a role similar to \(\lambda\) but is restricted to be non‑negative.
- \(g(x) = x+y-4\) : the constraint function. For an inequality \(g(x) \ge 0\), feasibility means \(g(x) \ge 0\).
- \(\nabla f = (4,4)\) at \(x^*\).
- \(\nabla g = (1,1)\).
- The equation \(\nabla f = \mu \nabla g\) yields \(\mu=4\).
- Complementary slackness: \(\mu \cdot g(x^*) = 4 \cdot (2+2-4) = 0\). This holds because the constraint is active (\(g=0\)).
- Dual feasibility: the computed multiplier \(\mu=4\) is non‑negative.

### Comparison of Constrained Optimisation Concepts

| Concept                           | Mathematical Role                                              | When Constraint is Active                         | Neural Network Analogy                                                                      |
|-----------------------------------|----------------------------------------------------------------|---------------------------------------------------|---------------------------------------------------------------------------------------------|
| Lagrange Multiplier (\(\lambda\)) | Enforces equality \(g(x)=0\); \(\nabla f = -\lambda \nabla g\) | Always (equality must hold)                       | Regularisation penalty term weight; normalisation constraints                               |
| KKT Multiplier (\(\mu\))          | Enforces inequality \(g(x)\ge0\); \(\mu\ge0\), \(\mu\,g(x)=0\) | \(\mu>0\) if \(g(x)=0\); \(\mu=0\) if \(g(x)>0\)  | Support vectors in SVM; sparsity‑promoting norms; active constraints in robust optimisation |
| Complementary Slackness           | \(\mu \cdot g(x)=0\); links active set and multiplier          | Distinguishes binding vs. non‑binding constraints | Identifying active weight bounds (e.g., clipping thresholds)                                |
| Feasible Region                   | Set of \(x\) satisfying all constraints                        | Shape dictated by constraints                     | Parameter space restrictions (e.g., \(\ell_1\) ball, probability simplex)                   |

## Combined Constraints

The final problem: minimise \(f(x,y,z) = (x-1)^2+(y-2)^2+(z-3)^2\) subject to equality \(x+y+z-6=0\) and inequality \(x-1 \ge 0\).

### Variable definitions

- \(x, y, z \in \mathbb{R}\) : optimisation variables.
- The objective is a sum of squared distances from a target point \((a_1,a_2,a_3) = (1,2,3)\). Its unconstrained minimum is exactly at that point with zero loss.
- Equality constraint: \(h(x,y,z) = x+y+z-6 = 0\). This requires the variables to sum to 6.
- Inequality constraint: \(g(x) = x-1 \ge 0\), i.e., \(x \ge 1\).
- The unconstrained minimiser \((1,2,3)\) already satisfies \(1+2+3=6\) and \(1\ge 1\); thus the constraints are all satisfied and the constrained optimum coincides with the unconstrained one.
- In a general neural network context, this situation arises when the optimal unconstrained weights happen to already respect added constraints (e.g., a normalisation layer followed by a sum‑to‑one constraint that is automatically fulfilled by a softmax).

# Summary of Visualisations

The code produces five plots:

1. **Unconstrained convergence on \((x-3)^2+(y-2)^2\)** – compares GD, steepest descent, and Newton’s method.
2. **Quasi‑Newton convergence on \(x^2+2y^2-2xy\)** – shows BFGS outperforming gradient methods on an ill‑conditioned landscape.
3. **Lagrange multipliers** – illustrates gradient alignment and the geometry of the smallest circle touching the line.
4. **KKT conditions** – highlights the feasible region, active boundary, and lists the four KKT conditions.
5. **Combined constraints** – demonstrates a case where the unconstrained solution satisfies all constraints.

Each visualisation reinforces that the choice of optimisation strategy depends on the loss surface geometry and constraints. For neural networks, these ideas are scaled up and approximated, but the fundamental dynamics remain identical.

# Conclusion

The `optimization_code.py` file is a compact educational toolkit that implements and contrasts classical optimisation algorithms. Every method shown has a direct conceptual link to optimisers used in training neural networks. A detailed understanding of the variables and equations—how a learning rate scales a gradient, how backtracking ensures sufficient decrease, how curvature from a Hessian accelerates convergence, and how multipliers encode constraint information—is essential for designing, debugging, and improving learning algorithms. The comparison tables and visualisations make these abstract concepts tangible and serve as a valuable reference for anyone studying optimisation in the context of machine learning.
