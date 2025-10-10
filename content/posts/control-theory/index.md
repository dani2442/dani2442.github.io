---
title: "Control Theory"
date: 2025-10-03
tags: ["machine learning", "gaussian processes", "probability", "stochastic processes", "kernel methods"]
author: "Daniel López Montero"
showToc: true
draft: true
description: "A concise introduction to controllability for linear time-invariant systems: Kalman rank condition, the controllability Gramian, and the minimum-energy control."
ShowWordCount: false
ShowReadingTime: true
comments: true
UseHugoToc: true
editPost:
    URL: "https://github.com/dani2442/dani2442.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---


Consider the linear time-invariant (LTI) system
$$
\dot x(t)=Ax(t)+Bu(t),\qquad x(t)\in\mathbb{R}^n,\; u(t)\in\mathbb{R}^m,
$$
with constant matrices $+A\in\mathbb{R}^{n\times n}$ and $B\in\mathbb{R}^{n\times m}$. The vector $x(t)$ is the system state and $u(t)$ is the control input we can apply.

Two basic questions arise:

- Existence: Given a target state $x_{\mathrm{target}}$ and a horizon $T>0$, does there exist a control $u(\cdot)$ that steers $x(0)=x_0$ to $x(T)=x_{\mathrm{target}}$? 
- Construction: If such a control exists, can we construct a simple, possibly optimal, control that achieves it?

For finite-dimensional LTI systems both questions have clean answers. The classical Kalman rank condition characterizes existence (controllability). When the condition holds one can also construct an explicit minimum-energy control using the controllability Gramian.

## 1. Kalman controllability criterion
The pair $(A,B)$ (or the LTI system above) is controllable (i.e., one can steer any initial state to any final state in finite time) if and only if the controllability matrix
$$
\mathcal{C} = \big[\,B\;\mid\; AB \;\mid\; A^2B \;\mid\; \dots \;\mid\; A^{n-1}B\,\big]
$$
has rank $n$. Here $\mathcal{C}$ is an $n\times nm$ matrix whose columns collect the directions available through $B$ and propagated by powers of $A$.

We sketch a proof of the equivalence and then show how to construct a minimum-energy control when the criterion holds.


The solution with initial condition $x(0)=x_0$ is
$$
x(T)=e^{AT}x_0 + \int_{0}^{T} e^{A(T-s)}B\,u(s)\,ds.
$$
Define the reachable set from $x_0$ at time $T$:
$$
\mathcal{R}_T(x_0)=\left\{ x(T) : x(T)=e^{AT}x_0+\int_0^T e^{A(T-s)}B\,u(s)\,ds,\; u(\cdot)\ \text{admissible}\right\}.
$$
It suffices to study reachability from the origin $x_0=0$, so write
$$
\mathcal{R}_T = \left\{\int_0^T e^{A(T-s)}B\,u(s)\,ds \right\}.
$$
The system is controllable if for some $T>0$ we have $\mathcal{R}_T=\mathbb{R}^n$.

> Direction 1: rank condition fails $\Rightarrow$ not controllable

If $\operatorname{rank}\mathcal{C}< n$, then there exists a nonzero vector $q\in\mathbb{R}^n$ such that

$$
q^T A^k B = 0,\qquad k=0,1,\dots,n-1.
$$
Since $e^{A\tau}=\sum_{k=0}^\infty \frac{\tau^k}{k!}A^k$, it follows that for every $\tau\ge0$,
$$
q^T e^{A\tau}B =\sum_{k=0}^\infty \frac{\tau^k}{k!}q^T A^k B = 0.
$$
Hence for every admissible $u(\cdot)$,
$$
q^T\int_0^T e^{A(T-s)}B\,u(s)\,ds = \int_0^T q^T e^{A(T-s)}B\,u(s)\,ds = 0.
$$
Therefore the scalar $q^T x(T)=q^T e^{AT}x_0$ is independent of the control; one cannot affect that component by any choice of $u$. The reachable set is a strict subset of $\mathbb{R}^n$, so the system is not controllable. This proves the contrapositive: controllability implies $\operatorname{rank}\mathcal{C}=n$.


> Direction 2: If $\operatorname{rank}(\mathcal{C}) = n$ then controllable

Define the finite-horizon controllability Gramian for $T>0$:
$$
W_c(T) \;=\; \int_0^T e^{A\tau} B B^T e^{A^T\tau}\, d\tau.
$$
Two facts are central:

1. For any desired displacement $d\in\mathbb{R}^n$ there exists an input $u(\cdot)$ steering $x(0)=0$ to $x(T)=d$ if and only if $d$ lies in the column space (image) of $W_c(T)$. In particular, if $W_c(T)$ is invertible (positive definite) then every $d$ is reachable at time $T$.

2. $W_c(T)$ is positive definite for some (equivalently, sufficiently large) $T>0$ if and only if $\operatorname{rank}\mathcal{C}=n$.

We outline why these hold and how to construct a minimum-energy control.


Let $d = x(T)-e^{AT}x_0$ be the desired displacement. We seek $u(\cdot)$ such that
$$
\int_0^T e^{A(T-s)}B\,u(s)\,ds = d.
$$
Among all controls achieving this, the minimum-energy control (minimizing $J(u)=\tfrac12\int_0^T \|u(s)\|^2 ds$) is obtained by calculus of variations / Lagrange multipliers. Define the Lagrangian
$$
\mathcal{L}(u,\lambda)=\tfrac12\int_0^T u(s)^T u(s)\,ds + \lambda^T\Big(\int_0^T e^{A(T-s)}B\,u(s)\,ds - d\Big).
$$
Stationarity with respect to $u$ (pointwise) gives
$$
u(s) + B^T e^{A^T(T-s)}\lambda = 0 \quad\Rightarrow\quad u(s) = -B^T e^{A^T(T-s)}\lambda.
$$
Plugging into the constraint yields
$$
d = -\int_0^T e^{A(T-s)}B B^T e^{A^T(T-s)}\,\lambda\,ds = -W_c(T)\lambda.
$$
Therefore, if $W_c(T)$ is invertible, $\lambda = -W_c(T)^{-1} d$ and the minimum-energy control is
$$
\boxed{\;u^*(s)=B^T e^{A^T(T-s)} W_c(T)^{-1} d\; }.
$$
Substituting this $u^*$ into the state equation yields the desired final state $x(T)=e^{AT}x_0+d$.

### Why invertibility of $W_c(T)$ is equivalent to the Kalman rank condition

If $\operatorname{rank}\mathcal{C}\lt n$ then, as shown earlier, there exists $q\neq0$ with $q^T A^k B=0$ for $k=0,\dots,n-1$. This implies $B^T e^{A^T\tau}q\equiv0$ and hence
$$
q^T W_c(T) q = \int_0^T \|B^T e^{A^T\tau}q\|^2\,d\tau = 0,
$$
so $W_c(T)$ is singular for every $T>0$.

Conversely, if $\operatorname{rank}\mathcal{C}=n$ but $W_c(T)$ were singular for every $T$, there would exist $q\neq0$ with $q^T W_c(T) q=0$ for all $T$. Hence $B^T e^{A^T\tau} q\equiv0$ for all $\tau\ge0$, and differentiating at $\tau=0$ repeatedly yields
$$
B^T (A^T)^k q = 0\qquad\text{for all }k\ge0,
$$
equivalently $q^T A^k B=0$ for all $k\ge0$. By Cayley–Hamilton only the first $n$ powers are independent, so this contradicts $\operatorname{rank}\mathcal{C}=n$. Therefore for some $T>0$ the Gramian $W_c(T)$ is invertible, and reachability follows from the construction in the previous section.

### Example: Spring-mass-control

Consider a mass $m$ attached to a spring with stiffness $k$ and a damper with damping coefficient $c$. Apply an external force $u(t)$ to the mass. Let $x_1$ be the position of the mass and $x_2$ the velocity of the mass. From Newton's Law,
$$m(dx_2/dt) = -kx_1 - cx_2 + u(t)$$
Writing $x=(x_1, x_2)^\top$ with $x_2 = \dot x_1$ yields
$$
    \begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -\frac{k}{m} & -\frac{c}{m} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} u
$$
To implement it in code, we need to integrate the solution using some integration scheme, such as Euler, Runge-Kutta, etc. I have opted to use `torchsde`, not because it is better, but due to the ability to add noise to the solution. There are other alternatives with the same framework such as `torchdiffeq`. Firstly, we need to implement the equation:
$$
dX_t = f(X_t, t) dt + g(X_t, t) dW_t = (AX_t + Bu_t)dt + \sigma dW_t
$$
```python
class LinearControlSDE(torch.nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'
    
    def __init__(self, A, B, control_input: callable = None, sigma=0.):
        super().__init__()
        self.A = A
        self.B = B
        self.sigma = sigma
        self.state_size = A.shape[0]
        self.control_size = B.shape[1]
        self.control_input = control_input
        
    def f(self, t, y):
        """Drift function: Ax + Bu"""
        dx = torch.matmul(y, self.A.T)
        u = self.control_input(t, y)
        dx += torch.matmul(u, self.B.T)
        return dx
    
    def g(self, t, y):
        """Diffusion function: σ"""
        batch_size = y.shape[0]
        state_diffusion = self.sigma * torch.ones(batch_size, self.state_size, device=y.device, dtype=y.dtype)
        return state_diffusion
```
And, for instance using a null control, it yields the following trajetory
```python
m, k, c = 1.0, 1.0, 0.5
A = torch.tensor([[0.0, 1.0],[-k/m, -c/m]])
B = torch.tensor([[0.0], [1.0/m]])

sde = LinearControlSDE(A, B, control_input=control_input, sigma=0.01)

# Initial condition: [position, velocity]
num_trajectories = 5
y0 = torch.tensor([2.0, 0.0]).repeat(num_trajectories, 1) + 0.1*torch.randn((num_trajectories,1))

# Time span and simulate
t_span = [0.0, 20.0]
ts = torch.linspace(t_span[0], t_span[1], 100)
ys = torchsde.sdeint(sde, y0, ts, method="euler", dt_min=1e-1)

plt.figure(figsize=(10, 8))
for i in range(num_trajectories):
    plt.plot(ys_np[:, i, 0], ys_np[:, i, 1], alpha=0.7, label=f'Trajectory {i+1}')
plt.scatter(y0[:,0].numpy(), y0[:,1].numpy(), s=20, c="k", label='Initial condition')
plt.xlabel('Position (x₁)')
plt.ylabel('Velocity (x₂)')
plt.title('Phase Portrait')
plt.grid(True)
plt.legend()
plt.savefig('phase_portrait_sde.png', dpi=150, bbox_inches='tight')
plt.show()
```


Now, let us implement the optimal control explained in the previous section. Firstly, we need to compute the Gramian matrix
```python
def compute_gramian(A, B, T, n_steps=200):
    """
    Compute controllability Gramian numerically with trapezoidal integration.
    Wc(T) = ∫_0^T e^{At} B B^T e^{A^T t} dt
    """
    n = A.shape[0]
    times = torch.linspace(0, T, n_steps + 1, dtype=A.dtype, device=A.device)
    
    # Compute all matrix exponentials at once using vmap
    # Shape: (n_steps+1, n, n)
    At = A.unsqueeze(0) * times.view(-1, 1, 1)
    Et = torch.vmap(torch.matrix_exp)(At)
    
    # Compute Mt = Et @ B @ B^T @ Et^T for all time steps
    # Shape: (n_steps+1, n, n)
    BBT = B @ B.T
    Mt = Et @ BBT @ Et.transpose(-2, -1)
    
    # Apply trapezoidal weights: 0.5 for first and last, 1.0 for middle
    weights = torch.ones(n_steps + 1, dtype=A.dtype, device=A.device)
    weights[0] = 0.5
    weights[-1] = 0.5
    
    # Weighted sum with broadcasting
    Wc = torch.sum(Mt * weights.view(-1, 1, 1), dim=0)
    Wc *= (T / n_steps)
    
    return Wc

class OptimalRoute(nn.Module):
    def __init__(self, A, B, x_target, T, n_steps=200, regularize=1e-9):
        super().__init__()
        self.x_target = x_target.reshape(1, -1)
        self.A = A
        self.B = B
        self.T = T
        self.m_exp = torch.matrix_exp(A * T)

        # Gramian
        Wc = compute_gramian(A, B, T, n_steps=n_steps)
        Wc_reg = Wc + regularize * torch.eye(A.shape[0])
        self.Wc_inv = torch.inverse(Wc_reg)

    def forward(self, t, y):
        s = self.T - t
        term = torch.matrix_exp(self.A.T * s)
        u = ((self.x_target - y @ self.m_exp.T) @ self.Wc_inv.T) @ term.T @ self.B
        return u
```



## 2. Learning the Dynamic of the Model

Let us continue to work on the same problem
$$
x' = Ax + Bu
$$
However, this time we have trajectories (data), but we do not know this time the concrete values $A$ and $B$ that describes it. So we may ask ourselves:

> Can we learn the dynamic of the system with this data? And how?