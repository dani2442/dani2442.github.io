---
title: "Continuous-time Reinforcement Learning: HJB and Policy Iteration"
date: 2026-01-25
tags: ["machine learning", "reinforcement learning", "control theory", "PDEs"]
categories: ["Reinforcement Learning"]
author: "Daniel López Montero"
showToc: true
draft: true
description: "Deriving the HJB equation, implementing neural policy iteration for continuous-time stochastic control, and validating on LQR and Merton's portfolio problem."
ShowWordCount: false
ShowReadingTime: true
comments: true
TocOpen: false
UseHugoToc: true
editPost:
    URL: "https://github.com/dani2442/dani2442.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

# 1. Introduction

If time is continuous and the system evolves according to an ODE/SDE, the RL problem is naturally written as a control problem:
$$
dX_t = f(X_t,a_t)\,dt + \Sigma(X_t,a_t)\,dW_t
$$
where $X_t$ is the system state, $a_t$ is the control action, $W_t$ is a standard Wiener process, and $f$ and $\Sigma$ define the system dynamics. The reward is given by a function $r(x,a)$, and the objective is to maximize the expected discounted reward over an infinite horizon:
$$
J(\pi):=\mathbb{E}\Big[\int_0^\infty e^{-\rho t}r(X_t,a_t)\,dt\Big],\qquad a_t\sim \pi(\cdot\mid X_t)
$$
where $\rho>0$ is the discount rate. The value function
$$
V(x):=\sup_\pi \mathbb{E}\Big[\int_0^\infty e^{-\rho t}r(X_t,a_t)\,dt \Big| X_0=x\Big]
$$
Under suitable regularity conditions:
1. $f(\cdot,a)$, $\Sigma(\cdot,a)$, $r(\cdot,a)$ are continuous in $(x,a)$; Lipschitz in $x$ uniformly in $a$.
2. $\Sigma\Sigma^\top(x,a)$ is bounded and uniformly nondegenerate (for classical $C^2$ theory; if you drop this you typically work in viscosity form).
3. $r$ is bounded (or has at most linear growth with enough integrability).
4. $V\in C^2(\mathbb R^d)$ and bounded (or polynomial growth, with the usual technical modifications).

> Then, the value function satisfies the *Hamilton–Jacobi–Bellman* (HJB) PDE:
> $$ \rho V(x)=\max_{a\in \mathcal{A}}\Big\{ r(x,a)+\mathcal{L}^a V(x)\Big\}.\tag{1} $$
> where $\mathcal{L}^a$ is the infinitesimal generator of the diffusion process under action $a$:
> $$ \mathcal{L}^a \varphi(x):=\nabla \varphi(x)^\top f(x,a)
+\tfrac12 \mathrm{Tr}\big(\Sigma\Sigma^\top \nabla^2 \varphi(x)\big) $$

*Proof sketch*: 
Fix $x$ and a small $h>0$. By the dynamic programming principle (DPP),
$$
V(x)=\sup_{\pi}\mathbb E_x\left[\int_{0}^{h}e^{-\rho t}r(X_t,a_t)dt + e^{-\rho h}V(X_h)\right].
$$
For the PDE derivation, it suffices to consider controls that hold a constant action $a$ over $[0,h]$ (then optimize over $a$ at the end). For such an $a$,
$$
V(x)=\sup_{a\in\mathcal A}\mathbb E_x\left[\int_{0}^{h}e^{-\rho t}r(X_t,a)dt + e^{-\rho h}V(X_h)\right].
$$

Use Taylor expansions as $h\to0$. Since $r$ is continuous and $X_t=x+o(1)$ over short times,
$$
  \mathbb E_x\left[\int_0^h e^{-\rho t} r(X_t,a)dt\right]= h\,r(x,a)+o(h).
$$

By Itô’s formula for $V(X_t)$ and the definition of the generator,
$$
  \mathbb E_x[V(X_h)] = V(x) + h\,\mathcal L^a V(x) + o(h),
$$
and $e^{-\rho h}=1-\rho h+o(h)$. Hence
$$
  \mathbb E_x[e^{-\rho h}V(X_h)] = V(x) + h\big(\mathcal L^a V(x)-\rho V(x)\big)+o(h).
$$

Plugging into the DPP:
$$
V(x)=\sup_{a}\Big\{V(x) + h\big(r(x,a)+\mathcal L^aV(x)-\rho V(x)\big)+o(h)\Big\}.
$$
Cancel $V(x)$, divide by $h$, and let $h\downarrow 0$:
$$
0=\sup_{a\in\mathcal A}\big\{r(x,a)+\mathcal L^aV(x)-\rho V(x)\big\} \quad \Leftrightarrow \quad
\rho V(x)=\max_{a\in\mathcal A}\left\{r(x,a)+\mathcal L^aV(x)\right\},
$$
which is exactly (1). $\quad\blacksquare$

---

The exact same argument can be used to derive the HJB for the non-autonomous case, where $f$, $\Sigma$, $r$ depend on time as well (see Appendix A).


Let us define the $Q$-function as
$$
Q(x,a):=\frac{1}{\rho} \Big(r(x,a)+\mathcal{L}^a V(x)\Big)\tag{2}
$$
Then if follows, by HJB (1), that 
$$V(x)=\max_{a\in \mathcal{A}} Q(x,a).\tag{3}$$
Combining (2) and (3), we get the following nonlinear equation for $Q$:
$$
\rho Q(x,a)=r(x,a)+\mathcal{L}^a \Big(\max_{a'\in \mathcal{A}} Q(x,a')\Big).\tag{4}
$$
Formally this is the direct continuous-time analogue of the discrete-time Bellman optimality equation. Note, however, that $x\mapsto \max_{a'}Q(x,a')$ need not be $C^2$ even if each $x\mapsto Q(x,a)$ is smooth. A standard way to make (4) mathematically precise is to interpret it in the viscosity sense (or to impose additional assumptions such as a unique maximizer with enough regularity).
This is the continuous-time analogue of the Bellman equation for $Q$-functions in discrete time.

## Example 1: Stochastic Linear-Quadratic Regulator

The LQR is the canonical continuous-time control problem with a closed-form solution — ideal for validating our algorithm.

**Dynamics** (additive noise):
$$dX_t = (\alpha X_t + \beta\, a_t)\,dt + \sigma\, dW_t$$

**Reward** (quadratic cost):
$$r(x,a) = -\tfrac{1}{2}(q\,x^2 + r_a\,a^2)$$

The HJB (1) for this problem admits a quadratic value function $V(x) = -\tfrac{1}{2}Px^2 - c$, where $P$ solves the **discounted algebraic Riccati equation**:
$$\rho P = q + 2\alpha P - \frac{\beta^2}{r_a}P^2, \qquad c = \frac{\sigma^2 P}{2\rho}$$
and the optimal policy is linear: $a^*(x) = -\frac{\beta}{r_a}Px$.

We use $\alpha=-0.5,\; \beta=1,\; q=1,\; r_a=0.1,\; \sigma=0.3,\; \rho=0.1$.

```python
class StochasticLQR(ControlProblem):
    def drift(self, x, a):
        return x @ A.T + a @ B.T          # f(x,a) = Ax + Ba

    def reward(self, x, a):
        return -0.5 * ((x @ Q * x).sum(-1, keepdim=True)
                      + (a @ R * a).sum(-1, keepdim=True))

# Solve and compare
P, c, K = solve_are(A, B, Q, R, D, rho=0.1)   # exact Riccati
solver  = PolicyIteration(problem, config)
history = solver.solve()                         # neural PI
```

The learned value function and policy closely match the analytical solution:

![LQR — Value function and policy](lqr_value_policy.png)

A sample optimal trajectory drives the state toward zero while the cumulative discounted reward plateaus:

![LQR — Trajectory and reward](lqr_trajectory.png)

Convergence diagnostics — the HJB residual drops by over an order of magnitude:

![LQR — Convergence](lqr_convergence.png)


## Example 2: Merton Portfolio / Consumption

Merton's problem is a classical stochastic control problem in mathematical finance with a known closed-form solution under CRRA utility.

**Dynamics**:
$$dX_t = \big(r_f + \pi_t(\mu - r_f) - k_t\big)X_t\,dt + \pi_t\,\sigma\,X_t\,dW_t$$
where $X_t$ is wealth, $\pi_t\in[0,1.5]$ is the risky-asset fraction, and $k_t = c_t/X_t \in [0.005, 0.20]$ is the consumption-to-wealth ratio.

**Reward** (CRRA utility, $\gamma=2$):
$$r(x,a) = \frac{(k\,x)^{1-\gamma}}{1-\gamma}$$

The HJB yields *constant* optimal controls:
$$\pi^* = \frac{\mu - r_f}{\gamma\,\sigma^2}, \qquad k^* = \frac{\rho - (1-\gamma)M}{\gamma}, \qquad M = r_f + \frac{(\mu-r_f)^2}{2\gamma\sigma^2}$$
and a power-law value $V^*(x) \propto x^{1-\gamma}$.

We use $r_f=0.03,\;\mu=0.08,\;\sigma=0.20,\;\gamma=2,\;\rho=0.05$.

```python
class MertonProblem(ControlProblem):
    def drift(self, x, a):                        # a = (pi, c_rate)
        pi, cr = a[:, 0:1], a[:, 1:2]
        return (self.r_f + pi*(self.mu - self.r_f) - cr) * x

    def diffusion(self, x, a):
        return (a[:, 0:1] * self.sigma * x).unsqueeze(-1)

    def reward(self, x, a):
        c = (a[:, 1:2] * x).clamp(min=1e-8)
        return c.pow(1 - self.gamma) / (1 - self.gamma)
```

The learned value function tracks the exact power-law solution, and the learned controls converge close to the analytical constants $\pi^*\approx 0.625$ and $k^*\approx 0.048$:

![Merton — Value function and policy](merton_value_policy.png)

A sample wealth trajectory under the learned policy, with its cumulative discounted reward:

![Merton — Trajectory and reward](merton_trajectory.png)

Convergence diagnostics:

![Merton — Convergence](merton_convergence.png)

---

### Q-learning in continuous time

We will explore the simplest scenario to illustrate the idea. Assume that we are in position $x_t$ at time $t$, and we take action $a_t$. Then we also know $dx_t$, the infinitesimal change in position at time $t$ and we know $\Sigma$ (constant or even null). Then, we can compute $\mathcal{L}^{a_t}(\max_{a'\in \mathcal{A}} Q(x_t,a'))$ and using (4)
$$
Q(x_t, a_t) \leftarrow Q(x_t, a_t) +\eta\left[ r(x_t,a_t) + \mathcal{L}^{a_t}\Big(\max_{a'\in \mathcal{A}} Q(x_t,a')\Big)  - \rho Q(x_t,a_t)\right]
$$
Notice the similarity with the discrete-time Q-learning update:
$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\eta\Big[r(s_t,a_t)+\gamma\max_{a'\in \mathcal{A}}Q(s_{t+1},a')-Q(s_t,a_t)\Big].
$$


## Appendix A: Non-autonomous case


Let the dynamics and reward depend on time:
$$
dX_t=f(t,X_t,a_t),dt+\Sigma(t,X_t,a_t)\,dW_t,\qquad r=r(t,x,a).
$$
Define the time-dependent value (starting at time $t$ in state $x$):
$$
V(t,x):=\sup_\pi \mathbb E\Big[\int_t^\infty e^{-\rho(s-t)} r(s,X_s,a_s),ds\ \Big|\ X_t=x\Big].
$$
Then the (time-dependent) generator is
$$
\mathcal L_t^a \varphi(x)=\nabla \varphi(x)^\top f(t,x,a)+\tfrac12\mathrm{Tr}\big(\Sigma\Sigma^\top(t,x,a)\nabla^2\varphi(x)\big),
$$
and the HJB becomes
$$
\rho V(t,x)=\max_{a\in\mathcal A}\Big\{r(t,x,a)+\partial_t V(t,x)+\mathcal L_t^a V(t,x)\Big\}.
$$
Equivalently,
$$
-\partial_t V(t,x)=\max_{a\in\mathcal A}\Big\{r(t,x,a)+\mathcal L_t^a V(t,x)\Big\}-\rho V(t,x).
$$

In the autonomous case, $V(t,x)$ is time-independent, so $\partial_t V=0$ and you recover (1).



## Appendix B: Kullback-Liebler HJB

A common KL-regularized continuous-time control formulation fixes a reference (prior) policy $\mu(\cdot\mid x)$ and introduces a temperature $\alpha>0$. The HJB equation becomes a pointwise maximization over action distributions $\pi(\cdot\mid x)$:
$$
\rho V_\alpha(x)=
\max_{\pi(\cdot\mid x)}
\left\{
\int_{\mathcal{A}}\pi(a\mid x)\big(r(x,a)+\mathcal{L}^aV_\alpha(x)\big)\,da
-
\alpha\,\mathrm{KL}\!\big(\pi(\cdot\mid x)\,\|\,\mu(\cdot\mid x)\big)
\right\}.
\tag{5}
$$

The maximization over $\pi$ is explicit and yields the log-sum-exp / softmax form:
$$
\rho V_\alpha(x)=
\alpha\log\!\left(
\int_{\mathcal{A}}\mu(a\mid x)\exp\!\Big(\tfrac{r(x,a)+\mathcal{L}^aV_\alpha(x)}{\alpha}\Big)\,da
\right).
\tag{6}
$$
> If $\mathcal{A}$ is finite and $\mu(\cdot\mid x)$ is uniform on $\mathcal{A}$, then
> $$ \rho V_\alpha(x)=
  \alpha\log\!\left(\sum_{a\in\mathcal{A}}\exp\!\Big(\tfrac{r(x,a)+\mathcal{L}^aV_\alpha(x)}{\alpha}\Big)\right)   -\alpha\log|\mathcal{A}|. $$
As $\alpha\downarrow 0$, the Laplace principle gives $\alpha\log\int \exp(\cdot/\alpha)\to \max(\cdot)$, so you recover the hard HJB:
  $$
  \rho V(x)=\max_{a\in\mathcal{A}}\{r(x,a)+\mathcal{L}^aV(x)\}.
  $$

The maximizing policy in (5) is
$$
\pi_\alpha^*(a\mid x)=
\frac{\mu(a\mid x)\exp\!\Big(\tfrac{r(x,a)+\mathcal{L}^aV_\alpha(x)}{\alpha}\Big)}
{\int_{\mathcal{A}}\mu(a'\mid x)\exp\!\Big(\tfrac{r(x,a')+\mathcal{L}^{a'}V_\alpha(x)}{\alpha}\Big)\,da'}.
\tag{7}
$$
This is the continuous-time analogue of softmax over “advantages”, with $\mu$ acting as a prior.

Keeping the same definition of $Q_\alpha$, but now with the soft value $V_\alpha$:
$$
Q_\alpha(x,a)
:=\frac{1}{\rho}\Big(r(x,a)+\mathcal{L}^aV_\alpha(x)\Big).\tag{8}
$$
Then the soft value aggregation becomes the exact analogue of $V(x)=\max_a Q(x,a)$:
$$
V_\alpha(x)=
\frac{\alpha}{\rho}\log\!\left(
\int_{\mathcal{A}}\mu(a\mid x)\exp\!\Big(\tfrac{\rho\,Q_\alpha(x,a)}{\alpha}\Big)\,da
\right).
\tag{9}
$$
Substituting (9) back into $\rho Q_\alpha(x,a)=r(x,a)+\mathcal{L}^aV_\alpha(x)$ yields the soft-HJB / KL Bellman equation for $Q_\alpha$:
$$
\rho Q_\alpha(x,a)=
r(x,a)
+
\mathcal{L}^a\left[
\frac{\alpha}{\rho}\log\!\left(
\int_{\mathcal{A}}\mu(a'\mid x)\exp\!\Big(\tfrac{\rho\,Q_\alpha(x,a')}{\alpha}\Big)\,da'
\right)
\right](x).
\tag{10}
$$
This is the direct “soft” replacement of the hard-max PDE (4).

### Expanding the generator term in (10)

Equation (10) is often read in the viscosity sense, since $x\mapsto V_\alpha(x)$ may fail to be $C^2$ without extra assumptions. If you do assume $V_\alpha\in C^2$ (and can differentiate under the integral defining $V_\alpha$), then $\mathcal{L}^a[V_\alpha](x)$ can be expanded explicitly.

Define the partition function
$$
Z(x):=\int_{\mathcal A}\mu(a'\mid x)\exp\!\Big(\tfrac{\rho}{\alpha}Q_\alpha(x,a')\Big)\,da',
\qquad
V_\alpha(x)=\frac{\alpha}{\rho}\log Z(x).
$$
The associated Boltzmann policy is
$$
\bar\pi_\alpha(a'\mid x):=\frac{\mu(a'\mid x)\exp\!\big(\tfrac{\rho}{\alpha}Q_\alpha(x,a')\big)}{Z(x)}.
$$

Let
$$
h(x,a'):=\log\mu(a'\mid x)+\frac{\rho}{\alpha}Q_\alpha(x,a').
$$
Then the standard log-partition identities give
$$
\begin{aligned}
\nabla V_\alpha(x)
&=\mathbb E_{\bar\pi_\alpha(\cdot\mid x)}\!\big[\nabla_x Q_\alpha(x,a')\big]
+\frac{\alpha}{\rho}\,\mathbb E_{\bar\pi_\alpha(\cdot\mid x)}\!\big[\nabla_x\log\mu(a'\mid x)\big],
\end{aligned}
$$
and
$$
\begin{aligned}
\nabla^2 V_\alpha(x)
&=\mathbb E_{\bar\pi_\alpha(\cdot\mid x)}\!\big[\nabla_x^2 Q_\alpha(x,a')\big]
+\frac{\alpha}{\rho}\,\mathbb E_{\bar\pi_\alpha(\cdot\mid x)}\!\big[\nabla_x^2\log\mu(a'\mid x)\big]\\
&\quad+\frac{\alpha}{\rho}\,\operatorname{Cov}_{\bar\pi_\alpha(\cdot\mid x)}\!\Big(\nabla_x\log\mu(a'\mid x)+\frac{\rho}{\alpha}\nabla_x Q_\alpha(x,a')\Big),
\end{aligned}
$$
where $\operatorname{Cov}(\cdot)$ is the covariance matrix of the indicated vector under $\bar\pi_\alpha(\cdot\mid x)$.

For the dynamics under action $a$,
$$
\mathcal{L}^a\varphi(x)=\nabla\varphi(x)^\top f(x,a)+\tfrac12\mathrm{Tr}\big(\Gamma(x,a)\nabla^2\varphi(x)\big),
\qquad \Gamma(x,a):=\Sigma(x,a)\Sigma(x,a)^\top,
$$
so
$$
\mathcal{L}^a V_\alpha(x)=f(x,a)^\top \nabla V_\alpha(x)+\tfrac12\mathrm{Tr}\big(\Gamma(x,a)\nabla^2 V_\alpha(x)\big).
$$
Substituting these into (10), you get an explicit (but still nonlinear) PDE involving expectations and covariances of $\nabla_x Q_\alpha$ and $\nabla_x^2 Q_\alpha$ under $\bar\pi_\alpha(\cdot\mid x)$.

#### Simplification when $\mu$ is independent of $x$

If $\mu(a\mid x)\equiv \mu(a)$, then $\nabla_x\log\mu=\nabla_x^2\log\mu=0$ and
$$
\begin{cases}
\nabla V_\alpha(x)=\mathbb E_{\bar\pi_\alpha(\cdot\mid x)}\!\big[\nabla_x Q_\alpha(x,a')\big],\\
\nabla^2 V_\alpha(x)=\mathbb E_{\bar\pi_\alpha(\cdot\mid x)}\!\big[\nabla_x^2 Q_\alpha(x,a')\big]
+\frac{\rho}{\alpha}\operatorname{Cov}_{\bar\pi_\alpha(\cdot\mid x)}\!\big(\nabla_x Q_\alpha(x,a')\big).
\end{cases}
$$
The covariance term is the extra contribution coming from the log-sum-exp; it only affects (10) through the second-order part of the generator (i.e. it matters when $\Sigma\neq 0$).

#### Practical evaluation

Given $Q_\alpha(x,\cdot)$ and its $x$-derivatives, the remaining objects are just expectations under $\bar\pi_\alpha(\cdot\mid x)$: sums when $\mathcal A$ is finite, and typically Monte Carlo / quadrature approximations when $\mathcal A$ is continuous. In parametric settings (e.g. neural networks), $\nabla_x Q_\alpha$ and $\nabla_x^2 Q_\alpha$ can be obtained by automatic differentiation (the Hessian being the expensive part).


### References

[1] Jia, Yanwei, and Xun Yu Zhou. "q-Learning in continuous time." Journal of Machine Learning Research 24, no. 161 (2023): 1-61.

[2] Jia, Yanwei, and Xun Yu Zhou. "Policy gradient and actor-critic learning in continuous time and space: Theory and algorithms." Journal of Machine Learning Research 23, no. 275 (2022): 1-50.

[3] Hamilton-Jacobi-Bellman Equations,
Stochastic Differential Equations by Benjamin Moll https://benjaminmoll.com/wp-content/uploads/2019/07/Lecture4_ECO521_web.pdf