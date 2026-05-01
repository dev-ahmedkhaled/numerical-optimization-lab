# Numerical Optimization Techniques in Python

This project implements and visualizes fundamental numerical optimization algorithms used in machine learning and mathematical optimization.

It combines:
- Theoretical derivations  
- Numerical implementations  
- Visual convergence analysis  

---

## Algorithms Implemented

### Unconstrained Optimization
- Gradient Descent (fixed step size)
- Steepest Descent (Armijo backtracking line search)
- Newton’s Method (second-order optimization)
- BFGS Quasi-Newton Method

### Constrained Optimization
- Lagrange Multipliers (equality constraints)
- Karush-Kuhn-Tucker (KKT) conditions (inequality constraints)
- Combined equality + inequality constraints

---

## Mathematical Problems Solved

### 1. Quadratic Function
f(x, y) = (x - 3)^2 + (y - 2)^2

### 2. Coupled Quadratic Function
f(x, y) = x^2 + 2y^2 - 2xy

### 3. Constrained Optimization
- Equality constraint: x + y = 4
- Inequality constraint: x + y >= 4

### 4. Multi-variable Constraint System
- Sum constraint: x1 + x2 + x3 = 6
- Bound constraint: xi >= 1

---

## Project Structure

.
├── optimization_code.py
├── outputs/
│   ├── unconstrained_convergence.png
│   ├── quasi_newton_convergence.png
└── README.md

---



## Key Insights

- Gradient Descent: simple but slow convergence  
- Steepest Descent: better direction via line search  
- Newton’s Method: very fast convergence (quadratic near optimum)  
- BFGS: approximates Hessian efficiently without second derivatives  
- KKT Conditions: general framework for constrained optimization  

---

## Libraries Used

- NumPy  
- SciPy  
- Matplotlib  

---

## How to Run

Install dependencies:

pip install numpy scipy matplotlib

Run the script:

python optimization_code.py

---

## Learning Outcome

This project demonstrates:
- How optimization algorithms behave geometrically  
- Difference between first-order and second-order methods  
- How constraints affect optimal solutions  
- Connection between theory (KKT/Lagrange) and implementation  

---

## Author

University project on Optimization Techniques  
Focused on implementing and visualizing classical optimization methods

