"""
Optimization Techniques - Python Implementation
Covers: Gradient Descent, Steepest Descent, Newton's Method, Quasi-Newton (BFGS),
        Lagrange Multipliers, KKT Conditions, Equality & Inequality Constraints

Generates 5 figures:
  1. unconstrained_convergence.png    - f(x,y)=(x-3)^2+(y-2)^2
  2. quasi_newton_convergence.png     - f(x,y)=x^2+2y^2-2xy
  3. lagrange_multipliers.png         - min x^2+y^2 s.t. x+y=4
  4. kkt_conditions.png               - min x^2+y^2 s.t. x+y>=4
  5. combined_constraints.png         - min sum(xi-ai)^2 s.t. sum(xi)=6, x>=1
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# UNCONSTRAINED METHODS
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

def steepest_descent(f, grad_f, x0, tol=1e-8, max_iter=1000):
    """Steepest Descent with Armijo backtracking line search."""
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        d = -g
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

def newtons_method(f, grad_f, hess_f, x0, tol=1e-8, max_iter=100):
    """Pure Newton's Method for unconstrained optimization."""
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess_f(x)
        d = np.linalg.solve(H, -g)
        x = x + d
        history.append(x.copy())
    return x, f(x), k+1, history

def bfgs(f, grad_f, x0, tol=1e-8, max_iter=200):
    """BFGS Quasi-Newton method."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    H = np.eye(n)
    history = [x.copy()]
    g = grad_f(x)
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        d = -H @ g
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
# CONSTRAINED METHODS (analytical + numerical)
# ============================================================

def lagrange_equality():
    """Minimize f(x,y)=x^2+y^2 subject to x+y=4."""
    lam = 4.0
    x_star = lam / 2
    y_star = lam / 2
    f_star = x_star**2 + y_star**2
    result = minimize(lambda xy: xy[0]**2 + xy[1]**2,
                      x0=[0.0, 0.0],
                      constraints={'type': 'eq', 'fun': lambda xy: xy[0]+xy[1]-4},
                      method='SLSQP')
    return (x_star, y_star, lam, f_star), result

def kkt_inequality():
    """Minimize f(x,y)=x^2+y^2 subject to x+y>=4."""
    mu = 4.0
    x_star, y_star = 2.0, 2.0
    f_star = x_star**2 + y_star**2
    g_star = x_star + y_star - 4
    cs = mu * g_star
    result = minimize(lambda xy: xy[0]**2 + xy[1]**2,
                      x0=[0.5, 0.5],
                      constraints={'type': 'ineq', 'fun': lambda xy: xy[0]+xy[1]-4},
                      method='SLSQP')
    return (x_star, y_star, mu, f_star, g_star, cs), result

def combined_constraints():
    """Minimize f(x,y,z)=(x-1)^2+(y-2)^2+(z-3)^2 s.t. x+y+z=6, x>=1."""
    result = minimize(lambda v: (v[0]-1)**2 + (v[1]-2)**2 + (v[2]-3)**2,
                      x0=[1.0, 1.0, 4.0],
                      constraints=[
                          {'type': 'eq',  'fun': lambda v: v[0]+v[1]+v[2]-6},
                          {'type': 'ineq','fun': lambda v: v[0]-1}
                      ],
                      method='SLSQP')
    return result

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_convergence(histories, labels, title, f, f_star=0.0, xlim=(-1,5), ylim=(-1,5)):
    """Plot unconstrained optimization paths + convergence history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold', color='#1E2761')

    # Contour
    x_vals = np.linspace(xlim[0], xlim[1], 300)
    y_vals = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(x_vals, y_vals)
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
                 color=colors[i % len(colors)], label=lbl, alpha=0.9, linewidth=2)
        ax1.plot(arr[0,0], arr[0,1], 'X', color=colors[i % len(colors)], 
                 markersize=12, markeredgecolor='black', markeredgewidth=1)
        ax1.plot(arr[-1,0], arr[-1,1], '*', color=colors[i % len(colors)], 
                 markersize=14, markeredgecolor='black', markeredgewidth=1)

    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_title('Optimization Paths')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_xlim(xlim); ax1.set_ylim(ylim)
    ax1.set_aspect('equal', adjustable='box')

    # Convergence
    for i, (hist, lbl) in enumerate(zip(histories, labels)):
        iters = np.arange(len(hist))
        fvals = np.array([f(h) for h in hist])
        err = np.abs(fvals - f_star)
        err_clipped = np.clip(err, 1e-16, None)
        ax2.semilogy(iters, err_clipped, 
                     linestyle=linestyles[i % len(linestyles)],
                     marker=markers[i % len(markers)], markersize=4,
                     markevery=max(1, len(hist)//15) if len(hist) > 15 else 1,
                     color=colors[i % len(colors)], label=lbl, linewidth=2)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('|f(x) - f*| [log scale]')
    ax2.set_title('Convergence History')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=1e-16)

    plt.tight_layout()
    return fig


def plot_lagrange():
    """Plot Lagrange Multipliers visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Lagrange Multipliers — min x²+y²  s.t.  x+y=4', 
                 fontsize=14, fontweight='bold', color='#1E2761')

    # Left: Contour + constraint line + gradients
    x = np.linspace(-1, 5, 300)
    y = np.linspace(-1, 5, 300)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    ax1.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.7)

    x_line = np.linspace(-1, 5, 100)
    y_line = 4 - x_line
    ax1.plot(x_line, y_line, 'r--', linewidth=2.5, label='Constraint: x+y=4')
    ax1.plot(2, 2, 'r*', markersize=18, markeredgecolor='black', markeredgewidth=1.5, 
             label='Optimum x*=(2,2), f*=8')

    ax1.annotate('', xy=(0.5, 0.5), xytext=(2, 2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(0.8, 1.0, '∇f = (4,4)', color='blue', fontsize=10, fontweight='bold')

    ax1.annotate('', xy=(3.5, 3.5), xytext=(2, 2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(3.2, 3.0, '∇g = (1,1)', color='red', fontsize=10, fontweight='bold')

    ax1.plot([2], [2], 'ko', markersize=8)
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_title('Contour Plot with Constraint')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_xlim(-1, 5); ax1.set_ylim(-1, 5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Right: Geometric interpretation
    ax2.set_xlim(-1, 5); ax2.set_ylim(-1, 5); ax2.set_aspect('equal')
    for r in [1, 2, 3, 4]:
        circle = plt.Circle((0, 0), r, fill=False, color='blue', alpha=0.5, linewidth=1.5)
        ax2.add_patch(circle)
    ax2.plot(x_line, y_line, 'r--', linewidth=2.5, label='x + y = 4')
    circle_tangent = plt.Circle((0, 0), np.sqrt(8), fill=False, color='blue', 
                               linewidth=3, linestyle='-', label='Tangent circle (r=√8)')
    ax2.add_patch(circle_tangent)
    ax2.plot(2, 2, 'r*', markersize=18, markeredgecolor='black', markeredgewidth=1.5)
    ax2.text(2.2, 2.3, 'x*=(2,2)\nf*=8', fontsize=11, fontweight='bold', color='darkred')
    ax2.plot([0, 2], [0, 2], 'b-', linewidth=1.5, alpha=0.7)
    ax2.text(0.8, 1.3, 'r = √8', color='blue', fontsize=10)
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    ax2.set_title('Geometric View: Smallest Circle Touching the Line')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_kkt():
    """Plot KKT Conditions visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('KKT Conditions — min x²+y²  s.t.  x+y ≥ 4', 
                 fontsize=14, fontweight='bold', color='#1E2761')

    # Left: Feasible region + contour
    x = np.linspace(-1, 5, 300)
    y = np.linspace(-1, 5, 300)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    ax1.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.7)

    x_line = np.linspace(-1, 5, 100)
    y_line = 4 - x_line
    ax1.plot(x_line, y_line, 'r-', linewidth=2.5, label='Boundary: x+y=4')

    xx = np.linspace(-1, 5, 100)
    yy_lower = 4 - xx
    ax1.fill_between(xx, yy_lower, 5*np.ones_like(xx), alpha=0.15, color='green', label='Feasible: x+y ≥ 4')
    ax1.fill_between(xx, -1*np.ones_like(xx), yy_lower, alpha=0.1, color='red', label='Infeasible: x+y < 4')

    ax1.plot(2, 2, 'r*', markersize=18, markeredgecolor='black', markeredgewidth=1.5,
             label='Optimum x*=(2,2)')
    ax1.plot(0, 0, 'ko', markersize=10, markerfacecolor='white', markeredgewidth=2,
             label='Unconstrained min (0,0) — INFEASIBLE')
    ax1.annotate('', xy=(0.8, 0.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
    ax1.text(0.3, 0.3, 'Violates\nx+y≥4', color='gray', fontsize=9)

    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_title('Feasible Region & Contours')
    ax1.legend(fontsize=8.5, loc='upper left')
    ax1.set_xlim(-1, 5); ax1.set_ylim(-1, 5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Right: KKT Conditions breakdown
    ax2.axis('off'); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.text(5, 9.5, 'KKT Conditions at x*=(2,2)', fontsize=13, fontweight='bold',
             ha='center', color='#1E2761')

    boxes = [
        (7.5, '1. STATIONARITY', '#E8F4FD', '#3498DB', '∇f = μ·∇g  →  (4,4) = 4·(1,1)  ✓'),
        (5.5, '2. PRIMAL FEASIBILITY', '#E8F8E8', '#2ECC71', 'g(x*) = x+y-4 = 2+2-4 = 0  ≥ 0  ✓'),
        (3.5, '3. DUAL FEASIBILITY', '#FFF3E0', '#E67E22', 'μ = 4  ≥ 0  ✓'),
        (1.5, '4. COMPLEMENTARY SLACKNESS', '#FCE4EC', '#E91E63', 'μ · g(x*) = 4 · 0 = 0  ✓'),
    ]
    for y_pos, title, facecolor, edgecolor, text in boxes:
        rect = plt.Rectangle((0.5, y_pos), 9, 1.5, fill=True, facecolor=facecolor, 
                            edgecolor=edgecolor, linewidth=2)
        ax2.add_patch(rect)
        ax2.text(5, y_pos+1.1, title, fontsize=11, fontweight='bold', ha='center', color=edgecolor)
        ax2.text(5, y_pos+0.6, text, fontsize=10, ha='center', family='monospace')

    ax2.text(5, 0.8, 'All 4 KKT conditions satisfied → x*=(2,2) is optimal', 
             fontsize=11, fontweight='bold', ha='center', color='darkgreen',
             bbox=dict(boxstyle='round', facecolor='#D5F5E3', edgecolor='green'))

    plt.tight_layout()
    return fig


def plot_combined():
    """Plot Combined Constraints visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Combined Constraints — min (x-1)²+(y-2)²+(z-3)²  s.t.  x+y+z=6,  x≥1',
                 fontsize=13, fontweight='bold', color='#1E2761')

    # Left: 2D projection
    ax1.set_xlim(-0.5, 4); ax1.set_ylim(-0.5, 4); ax1.set_aspect('equal')
    x_pts = np.linspace(0, 4, 50)
    y_pts = np.linspace(0, 4, 50)
    X_pts, Y_pts = np.meshgrid(x_pts, y_pts)
    Z_pts = 6 - X_pts - Y_pts
    F_plane = (X_pts-1)**2 + (Y_pts-2)**2 + (Z_pts-3)**2
    valid = Z_pts >= 0
    ax1.contour(X_pts, Y_pts, np.where(valid, F_plane, np.nan), levels=15, cmap='viridis', alpha=0.7)

    ax1.axvline(x=1, color='red', linewidth=2.5, linestyle='--', label='x = 1 (boundary)')
    ax1.fill_betweenx([-0.5, 4], 1, 4, alpha=0.15, color='green', label='Feasible: x ≥ 1')
    ax1.plot(1, 2, 'r*', markersize=18, markeredgecolor='black', markeredgewidth=1.5,
             label='Optimum: x*=(1,2,3), f*=0')
    ax1.plot(1, 2, 'go', markersize=10, markerfacecolor='white', markeredgewidth=2,
             label='Same point! (unconstrained min satisfies constraints)')

    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_title('Projection onto x-y Plane (z = 6-x-y)')
    ax1.legend(fontsize=8.5, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Right: Explanation
    ax2.axis('off'); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.text(5, 9.5, 'Why is the optimum at (1,2,3)?', fontsize=13, fontweight='bold',
             ha='center', color='#1E2761')

    explanations = [
        (8.5, 'Unconstrained minimum of f is at (1,2,3)'),
        (7.8, 'because each squared term is zero there.'),
        (7.1, ''),
        (6.4, 'Check equality constraint:'),
        (5.7, 'x + y + z = 1 + 2 + 3 = 6  ✓'),
        (5.0, ''),
        (4.3, 'Check inequality constraint:'),
        (3.6, 'x = 1 ≥ 1  ✓ (active on boundary)'),
        (2.9, ''),
        (2.2, 'Since the unconstrained minimum'),
        (1.5, 'satisfies ALL constraints,'),
        (0.8, 'it is ALSO the constrained optimum!'),
    ]
    for y, text in explanations:
        bold = 'unconstrained' in text or 'satisfies' in text or 'ALSO' in text
        ax2.text(5, y, text, fontsize=10.5, ha='center', fontweight='bold' if bold else 'normal')

    rect = plt.Rectangle((0.5, 0.3), 9, 1.2, fill=True, facecolor='#D5F5E3',
                          edgecolor='green', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(5, 0.9, 'μ = 0 (constraint inactive), λ = 0', fontsize=11, 
             fontweight='bold', ha='center', color='darkgreen')

    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    x0 = [0.0, 0.0]

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
    print_section("4. QUASI-NEWTON (BFGS)  [f(x,y) = x²+2y²-2xy]")
    xopt2, fopt2, nit2, hist_bfgs = bfgs(f2, grad_f2, [2.0, 1.0])
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
    print("  Analytical: x*=(1,2,3), f*=0")

    # ---- GENERATE ALL PLOTS ----
    print("\n" + "="*60)
    print("GENERATING ALL PLOTS")
    print("="*60)

    # Plot 1: Unconstrained f1
    _, _, _, hist_gd1 = gradient_descent(f1, grad_f1, [0.0, 0.0], alpha=0.1)
    _, _, _, hist_sd1 = steepest_descent(f1, grad_f1, [0.0, 0.0])
    _, _, _, hist_nm1 = newtons_method(f1, grad_f1, hess_f1, [0.0, 0.0])
    fig1 = plot_convergence(
        [hist_gd1, hist_sd1, hist_nm1],
        ['Gradient Descent', 'Steepest Descent', "Newton's Method"],
        "Unconstrained Optimization — f(x,y) = (x-3)²+(y-2)²",
        f1, f_star=0.0, xlim=(-0.5, 4), ylim=(-0.5, 3.5)
    )
    fig1.savefig("unconstrained_convergence.png", dpi=150, bbox_inches='tight')
    print("  [1/5] unconstrained_convergence.png")

    # Plot 2: Unconstrained f2
    _, _, _, hist_gd2 = gradient_descent(f2, grad_f2, [2.0, 1.0], alpha=0.2)
    _, _, _, hist_sd2 = steepest_descent(f2, grad_f2, [2.0, 1.0])
    _, _, _, hist_nm2 = newtons_method(f2, grad_f2, hess_f2, [2.0, 1.0])
    _, _, _, hist_bfgs2 = bfgs(f2, grad_f2, [2.0, 1.0])
    fig2 = plot_convergence(
        [hist_gd2, hist_sd2, hist_nm2, hist_bfgs2],
        ['Gradient Descent', 'Steepest Descent', "Newton's Method", 'BFGS'],
        "Unconstrained Optimization — f(x,y) = x²+2y²-2xy",
        f2, f_star=0.0, xlim=(-0.5, 2.5), ylim=(-0.5, 1.5)
    )
    fig2.savefig("quasi_newton_convergence.png", dpi=150, bbox_inches='tight')
    print("  [2/5] quasi_newton_convergence.png")

    # Plot 3: Lagrange Multipliers
    fig3 = plot_lagrange()
    fig3.savefig("lagrange_multipliers.png", dpi=150, bbox_inches='tight')
    print("  [3/5] lagrange_multipliers.png")

    # Plot 4: KKT Conditions
    fig4 = plot_kkt()
    fig4.savefig("kkt_conditions.png", dpi=150, bbox_inches='tight')
    print("  [4/5] kkt_conditions.png")

    # Plot 5: Combined Constraints
    fig5 = plot_combined()
    fig5.savefig("combined_constraints.png", dpi=150, bbox_inches='tight')
    print("  [5/5] combined_constraints.png")

    print("\n" + "="*60)
    print("  ALL 5 PLOTS SAVED SUCCESSFULLY")
    print("="*60 + "\n")
