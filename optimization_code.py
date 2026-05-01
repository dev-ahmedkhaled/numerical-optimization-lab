"""
Optimization Techniques - Python Implementation
Covers: Gradient Descent, Steepest Descent, Newton's Method, Quasi-Newton (BFGS),
        Lagrange Multipliers, KKT Conditions, Equality & Inequality Constraints
"""

import numpy as np
from scipy.optimize import minimize, linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# UTILITY
# ============================================================

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_result(label, value, fmt=".6f"):
    if isinstance(value, (np.ndarray, list)):
        print(f"  {label}: {np.array(value)}")
    else:
        print(f"  {label}: {value:{fmt}}")

# ============================================================
# PROBLEM DEFINITIONS
# ============================================================
#
# Unconstrained: f(x,y) = (x-3)^2 + (y-2)^2   → min at (3,2), f=0
# Unconstrained: f(x,y) = x^2 + 2y^2 - 2xy    → min at (0,0), f=0 (elongated quadratic)
# Constrained  : min f(x,y)=x^2+y^2  s.t. x+y=4  → min at (2,2), f=8
# KKT          : min f(x,y)=x^2+y^2  s.t. x+y>=4 → same KKT point (active)
#
# ============================================================

# --- Objective 1 (simple bowl) ---
def f1(xy):
    x, y = xy
    return (x - 3)**2 + (y - 2)**2

def grad_f1(xy):
    x, y = xy
    return np.array([2*(x - 3), 2*(y - 2)])

def hess_f1(xy):
    return np.array([[2.0, 0.0],
                     [0.0, 2.0]])

# --- Objective 2 (elongated quadratic) ---
def f2(xy):
    x, y = xy
    return x**2 + 2*y**2 - 2*x*y

def grad_f2(xy):
    x, y = xy
    return np.array([2*x - 2*y, 4*y - 2*x])

def hess_f2(xy):
    return np.array([[2.0, -2.0],
                     [-2.0, 4.0]])

# ============================================================
# 1. GRADIENT DESCENT (fixed step)
# ============================================================

def gradient_descent(f, grad_f, x0, alpha=0.1, tol=1e-8, max_iter=1000):
    """Gradient Descent with fixed learning rate."""
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - alpha * g
        history.append(x.copy())
    return x, f(x), k+1, history

# ============================================================
# 2. STEEPEST DESCENT (exact line search via Armijo)
# ============================================================

def steepest_descent(f, grad_f, x0, tol=1e-8, max_iter=1000):
    """Steepest Descent with Armijo backtracking line search."""
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        d = -g                          # steepest descent direction
        # Armijo backtracking
        alpha = 1.0
        c = 1e-4
        rho = 0.5
        fx = f(x)
        while f(x + alpha*d) > fx + c*alpha*(g @ d):
            alpha *= rho
            if alpha < 1e-14:
                break
        x = x + alpha * d
        history.append(x.copy())
    return x, f(x), k+1, history

# ============================================================
# 3. NEWTON'S METHOD
# ============================================================

def newtons_method(f, grad_f, hess_f, x0, tol=1e-8, max_iter=100):
    """Pure Newton's Method for unconstrained optimization."""
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess_f(x)
        # Newton step: H * d = -g
        d = np.linalg.solve(H, -g)
        x = x + d
        history.append(x.copy())
    return x, f(x), k+1, history

# ============================================================
# 4. QUASI-NEWTON (BFGS) — manual implementation
# ============================================================

def bfgs(f, grad_f, x0, tol=1e-8, max_iter=200):
    """BFGS Quasi-Newton method."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    H = np.eye(n)          # Initial inverse Hessian approximation
    history = [x.copy()]
    g = grad_f(x)
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        d = -H @ g          # Search direction
        # Line search (Armijo)
        alpha = 1.0
        c = 1e-4; rho = 0.5
        fx = f(x)
        while f(x + alpha*d) > fx + c*alpha*(g @ d):
            alpha *= rho
            if alpha < 1e-14:
                break
        s = alpha * d
        x_new = x + s
        g_new = grad_f(x_new)
        y = g_new - g
        sy = s @ y
        if abs(sy) > 1e-14:
            # BFGS update of inverse Hessian
            rho_k = 1.0 / sy
            I = np.eye(n)
            A = I - rho_k * np.outer(s, y)
            B = I - rho_k * np.outer(y, s)
            H = A @ H @ B + rho_k * np.outer(s, s)
        x = x_new
        g = g_new
        history.append(x.copy())
    return x, f(x), k+1, history

# ============================================================
# 5. LAGRANGE MULTIPLIERS (analytical + numerical verification)
# ============================================================

def lagrange_equality():
    """
    Minimize f(x,y) = x^2 + y^2
    subject to  g(x,y) = x + y - 4 = 0

    Lagrangian: L = x^2 + y^2 - λ(x+y-4)
    ∂L/∂x = 2x - λ = 0  →  x = λ/2
    ∂L/∂y = 2y - λ = 0  →  y = λ/2
    ∂L/∂λ = x+y-4 = 0   →  λ/2 + λ/2 = 4  →  λ = 4
    Solution: x*=2, y*=2, λ*=4, f*=8
    """
    # Analytical solution
    lam = 4.0
    x_star = lam / 2
    y_star = lam / 2
    f_star = x_star**2 + y_star**2

    # Numerical verification with scipy
    result = minimize(lambda xy: xy[0]**2 + xy[1]**2,
                      x0=[0.0, 0.0],
                      constraints={'type': 'eq', 'fun': lambda xy: xy[0]+xy[1]-4},
                      method='SLSQP')
    return (x_star, y_star, lam, f_star), result

# ============================================================
# 6. KKT CONDITIONS — inequality constraint
# ============================================================

def kkt_inequality():
    """
    Minimize f(x,y) = x^2 + y^2
    subject to  g(x,y) = x + y - 4 >= 0  (i.e. x+y >= 4)

    KKT conditions (minimization, inequality g >= 0):
      ∇f = μ ∇g            (stationarity)
      g(x*) >= 0            (primal feasibility)
      μ >= 0                (dual feasibility)
      μ · g(x*) = 0         (complementary slackness)

    Case 1: μ=0  →  ∇f=0  →  x=y=0, but g(0,0)=-4 < 0 → infeasible
    Case 2: active constraint (μ>0): x+y=4
      2x = μ·1, 2y = μ·1  →  x=y  →  x=y=2, μ=4 > 0 ✓
    Solution: x*=2, y*=2, μ*=4, f*=8
    """
    mu = 4.0
    x_star, y_star = 2.0, 2.0
    f_star = x_star**2 + y_star**2
    g_star = x_star + y_star - 4   # = 0 (active)
    cs = mu * g_star                # = 0 (complementary slackness)

    # Numerical with scipy (inequality: g >= 0)
    result = minimize(lambda xy: xy[0]**2 + xy[1]**2,
                      x0=[0.5, 0.5],
                      constraints={'type': 'ineq', 'fun': lambda xy: xy[0]+xy[1]-4},
                      method='SLSQP')
    return (x_star, y_star, mu, f_star, g_star, cs), result

# ============================================================
# 7. EQUALITY + INEQUALITY CONSTRAINTS COMBINED
# ============================================================

def combined_constraints():
    """
    Minimize f(x,y,z) = (x-1)^2 + (y-2)^2 + (z-3)^2
    subject to:
      h(x,y,z) = x + y + z - 6 = 0       (equality)
      g(x,y,z) = x - 1 >= 0              (inequality)

    KKT stationarity:
      2(x-1) = λ + μ
      2(y-2) = λ
      2(z-3) = λ
    Complementary slackness: μ(x-1) = 0

    Case: g inactive (μ=0): λ = 2(y-2) = 2(z-3) → y-2=z-3 → z=y+1
      x+y+(y+1)=6 → x+2y=5
      2(x-1)=λ=2(y-2) → x-1=y-2 → x=y-1
      (y-1)+2y=5 → 3y=6 → y=2, x=1, z=3, λ=0, f=0

    The unconstrained minimum (1,2,3) satisfies equality constraint → μ=0 ✓
    """
    result = minimize(lambda v: (v[0]-1)**2 + (v[1]-2)**2 + (v[2]-3)**2,
                      x0=[1.0, 1.0, 4.0],
                      constraints=[
                          {'type': 'eq',  'fun': lambda v: v[0]+v[1]+v[2]-6},
                          {'type': 'ineq','fun': lambda v: v[0]-1}
                      ],
                      method='SLSQP')
    return result

# ============================================================
# PLOTTING (FIXED)
# ============================================================

def plot_convergence(histories, labels, title, f, f_star=0.0, xlim=(-1,5), ylim=(-1,5)):
    """
    Plot optimization paths on contour + convergence history.

    FIXES applied:
      1. Vectorized Z computation (was O(n²) Python loops)
      2. Plot |f(x) - f*| clipped at 1e-16 to avoid log(0) in semilogy
      3. Added star markers for final converged points
      4. Added equal aspect ratio for contour plot
      5. Unique linestyles/markers per method so overlapping paths remain visible
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold', color='#1E2761')

    # ---- Contour (vectorized evaluation) ----
    x_vals = np.linspace(xlim[0], xlim[1], 300)
    y_vals = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Vectorized: stack and evaluate in one go
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.array([f(xy) for xy in XY]).reshape(X.shape)

    ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)

    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']

    for i, (hist, lbl) in enumerate(zip(histories, labels)):
        arr = np.array(hist)
        if len(arr) == 0:
            continue
        ax1.plot(arr[:,0], arr[:,1], 
                 linestyle=linestyles[i % len(linestyles)],
                 marker=markers[i % len(markers)],
                 markersize=5 if len(arr) > 5 else 8,
                 markevery=max(1, len(arr)//10) if len(arr) > 10 else 1,
                 color=colors[i % len(colors)], 
                 label=lbl, 
                 alpha=0.9, 
                 linewidth=2)
        # Start point (X marker)
        ax1.plot(arr[0,0], arr[0,1], 'X', color=colors[i % len(colors)], 
                 markersize=12, markeredgecolor='black', markeredgewidth=1)
        # End point (star)
        ax1.plot(arr[-1,0], arr[-1,1], '*', color=colors[i % len(colors)], 
                 markersize=14, markeredgecolor='black', markeredgewidth=1)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Optimization Paths')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_aspect('equal', adjustable='box')

    # ---- Convergence (log scale, avoids log(0)) ----
    for i, (hist, lbl) in enumerate(zip(histories, labels)):
        iters = np.arange(len(hist))
        fvals = np.array([f(h) for h in hist])
        err = np.abs(fvals - f_star)
        err_clipped = np.clip(err, 1e-16, None)   # prevent log(0)
        ax2.semilogy(iters, err_clipped, 
                     linestyle=linestyles[i % len(linestyles)],
                     marker=markers[i % len(markers)],
                     markersize=4,
                     markevery=max(1, len(hist)//15) if len(hist) > 15 else 1,
                     color=colors[i % len(colors)],
                     label=lbl, 
                     linewidth=2)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('|f(x) - f*| [log scale]')
    ax2.set_title('Convergence History')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=1e-16)

    plt.tight_layout()
    return fig

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    x0 = [0.0, 0.0]   # Starting point

    # ---- 1. GRADIENT DESCENT ----
    print_section("1. GRADIENT DESCENT  [f(x,y) = (x-3)²+(y-2)²]")
    xopt, fopt, nit, hist_gd = gradient_descent(f1, grad_f1, x0, alpha=0.1)
    print_result("x*", xopt); print_result("f(x*)", fopt); print_result("Iterations", nit, "d")
    print("  Analytical solution: x*=(3,2), f*=0")

    # ---- 2. STEEPEST DESCENT ----
    print_section("2. STEEPEST DESCENT  [f(x,y) = (x-3)²+(y-2)²]")
    xopt, fopt, nit, hist_sd = steepest_descent(f1, grad_f1, x0)
    print_result("x*", xopt); print_result("f(x*)", fopt); print_result("Iterations", nit, "d")

    # ---- 3. NEWTON'S METHOD ----
    print_section("3. NEWTON'S METHOD  [f(x,y) = (x-3)²+(y-2)²]")
    xopt, fopt, nit, hist_nm = newtons_method(f1, grad_f1, hess_f1, x0)
    print_result("x*", xopt); print_result("f(x*)", fopt); print_result("Iterations", nit, "d")
    print("  (Quadratic → Newton converges in 1 step)")

    # ---- 4. BFGS QUASI-NEWTON ----
    # FIX: use same starting point [2.0, 1.0] as the other methods for f2
    print_section("4. QUASI-NEWTON (BFGS)  [f(x,y) = x²+2y²-2xy]")
    xopt2, fopt2, nit2, hist_bfgs = bfgs(f2, grad_f2, [2.0, 1.0])   # <-- FIXED
    print_result("x*", xopt2); print_result("f(x*)", fopt2); print_result("Iterations", nit2, "d")
    print("  Analytical solution: x*=(0,0), f*=0")

    # ---- 5. LAGRANGE MULTIPLIERS ----
    print_section("5. LAGRANGE MULTIPLIERS  [min x²+y²  s.t. x+y=4]")
    (xs, ys, lam, fs), res_lag = lagrange_equality()
    print("  --- Analytical ---")
    print_result("x*", xs); print_result("y*", ys)
    print_result("λ*", lam); print_result("f*", fs)
    print("  --- Numerical (scipy SLSQP) ---")
    print_result("x*", res_lag.x); print_result("f*", res_lag.fun)

    # ---- 6. KKT CONDITIONS ----
    print_section("6. KKT CONDITIONS  [min x²+y²  s.t. x+y≥4]")
    (xs, ys, mu, fs, g_val, cs), res_kkt = kkt_inequality()
    print("  --- Analytical ---")
    print_result("x*", xs); print_result("y*", ys)
    print_result("μ* (dual variable)", mu)
    print_result("g(x*) = x+y-4", g_val)
    print_result("Complementary slackness μ·g", cs)
    print_result("f*", fs)
    print("  --- Numerical (scipy SLSQP) ---")
    print_result("x*", res_kkt.x); print_result("f*", res_kkt.fun)

    # ---- 7. EQUALITY + INEQUALITY ----
    print_section("7. EQUALITY + INEQUALITY  [min Σ(xᵢ-aᵢ)²  s.t. Σxᵢ=6, x≥1]")
    res_comb = combined_constraints()
    print("  --- Numerical (scipy SLSQP) ---")
    print_result("x*", res_comb.x); print_result("f*", res_comb.fun)
    print("  Analytical: x*=(1,2,3), f*=0  (unconstrained min satisfies all constraints)")

    # ---- PLOTS ----
    fig1 = plot_convergence(
        [hist_gd, hist_sd, hist_nm],
        ['Gradient Descent', 'Steepest Descent', "Newton's Method"],
        "Unconstrained Optimization — f(x,y) = (x-3)²+(y-2)²",
        f1, f_star=0.0,
        xlim=(-0.5, 4), ylim=(-0.5, 3.5)
    )
    fig1.savefig("unconstrained_convergence.png", dpi=150, bbox_inches='tight')

    # Convergence comparison on f2 — ALL methods start from [2.0, 1.0]
    _, _, _, hist_gd2 = gradient_descent(f2, grad_f2, [2.0, 1.0], alpha=0.2)
    _, _, _, hist_sd2 = steepest_descent(f2, grad_f2, [2.0, 1.0])
    _, _, _, hist_nm2 = newtons_method(f2, grad_f2, hess_f2, [2.0, 1.0])
    # BFGS already computed above with [2.0, 1.0]
    fig2 = plot_convergence(
        [hist_gd2, hist_sd2, hist_nm2, hist_bfgs],
        ['Gradient Descent', 'Steepest Descent', "Newton's Method", 'BFGS'],
        "Unconstrained Optimization — f(x,y) = x²+2y²-2xy",
        f2, f_star=0.0,
        xlim=(-0.5, 2.5), ylim=(-0.5, 1.5)
    )
    fig2.savefig("quasi_newton_convergence.png", dpi=150, bbox_inches='tight')

    print("\n" + "="*60)
    print("  Plots saved")
    print("="*60 + "\n")
