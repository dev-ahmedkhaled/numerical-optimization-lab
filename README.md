# Numerical Optimization Techniques in Python

This project is a comprehensive implementation and visualization of classical optimization algorithms and constrained optimization theory using Python.

It covers both **theoretical derivations** and **numerical implementations**, making it suitable for understanding how optimization methods behave in practice.

---

## 📌 Features

### 🔹 Unconstrained Optimization Methods
- Gradient Descent (fixed step size)
- Steepest Descent (Armijo backtracking line search)
- Newton's Method
- Quasi-Newton Method (BFGS implementation)

### 🔹 Constrained Optimization
- Lagrange Multipliers (equality constraints)
- Karush-Kuhn-Tucker (KKT) conditions (inequality constraints)
- Combined equality + inequality constraints

### 🔹 Visualization
- Contour plots of objective functions
- Optimization paths of different methods
- Convergence plots (log-scale error reduction)

---

## 📊 Problems Solved

1. Quadratic bowl function:
   \[
   f(x,y) = (x-3)^2 + (y-2)^2
   \]

2. Coupled quadratic function:
   \[
   f(x,y) = x^2 + 2y^2 - 2xy
   \]

3. Constrained optimization:
   - Equality: \( x + y = 4 \)
   - Inequality: \( x + y \ge 4 \)

4. Combined constrained system:
   - \( \sum x_i = 6 \)
   - \( x_i \ge 1 \)

---

## 🧮 Libraries Used

- NumPy
- SciPy (`scipy.optimize.minimize`)
- Matplotlib

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install numpy scipy matplotlib
