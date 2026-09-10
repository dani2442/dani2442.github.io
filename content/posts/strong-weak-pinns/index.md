---
title: "Strong and Weak PINNs: Conditioning, Norms, and Finite Tests"
date: 2026-09-10
tags: ["mathematics", "deep learning", "PDE", "PINNs"]
categories: ["machine learning", "pde"]
author: "Daniel López Montero"
showToc: true
draft: true
description: "What weak residuals change in PINN optimization, what finite tests can miss, and a reproducible Poisson experiment."
ShowWordCount: false
ShowReadingTime: true
comments: true
TocOpen: false
UseHugoToc: true
---

A physics-informed neural network approximates a PDE solution by minimizing a residual loss. The **strong formulation** evaluates the differential equation directly. A **weak formulation** tests the equation against functions and can move derivatives from the network onto those tests through integration by parts. These are the starting points of [PINNs](https://maziarraissi.github.io/PINNs/) and [variational PINNs](https://arxiv.org/abs/1912.00873), respectively.

Two separate choices are hiding in that description: **which norm measures the residual**, and **how much of that residual is observed**. The full weak residual measures the PDE defect in a weaker topology; finitely many weak tests can additionally leave entire error directions undetected. The first choice explains the potential optimization benefit, the second explains why a very small training loss can be meaningless.

We make both statements exact for Poisson, then compare neural training and ablate the number and weighting of test functions. The [code and saved measurements](https://github.com/dani2442/dani2442_code/tree/main/strong-weak-pinns) reproduce every figure and number below. For an introduction to PINNs, see the [earlier post]({{< ref "/posts/pinns" >}}).

## 1. Strong and weak residuals

Let $\Omega\subset\mathbb R^d$ be a bounded Lipschitz domain, and consider

$$
-\Delta u^\star=f\quad\text{in }\Omega,
\qquad u^\star=0\quad\text{on }\partial\Omega,
\qquad f\in L^2(\Omega).
$$

Set $V=H_0^1(\Omega)$, with **energy inner product**

$$
(u,v)_V=\int_\Omega\nabla u\cdot\nabla v\,dx,
\qquad \|v\|_V=\|\nabla v\|_{L^2}.
$$

Poincaré's inequality makes this a Hilbert norm, equivalent to the usual $H^1$ norm on $V$. Assume the approximation $u_\theta\in V$ satisfies the boundary condition exactly, and write $e=u_\theta-u^\star$.

> **The metric used in every figure.** Unqualified, *energy* always means this norm, and the **relative energy error**
> $$\varepsilon_1=\frac{\|e\|_V}{\|u^\star\|_V}=\frac{\|\nabla u_\theta-\nabla u^\star\|_{L^2}}{\|\nabla u^\star\|_{L^2}}$$
> is the common accuracy metric every plot below reports. Since $u_\theta\equiv0$ gives $\varepsilon_1=1$, a value above $100\%$ means the approximation carries *more* energy error than the zero function.

**Strong loss.** If $\Delta u_\theta\in L^2$, define

$$
r_\theta=-\Delta u_\theta-f=-\Delta e,
\qquad
\mathcal L_s(u_\theta)=\frac12\|r_\theta\|_{L^2}^2.
$$

Taking $u_\theta\in H^2\cap H_0^1$ suffices; pointwise implementations use a sufficiently smooth network with spatial quadrature or collocation.

**Weak loss.** For any $u_\theta\in V$, define the functional

$$
\mathcal R_\theta(v)
=\int_\Omega\nabla u_\theta\cdot\nabla v\,dx
-\int_\Omega fv\,dx
=(e,v)_V,
\qquad v\in V.
$$

The residual $\mathcal R_\theta\in V'$ is a **linear functional**, not a single scalar. Its dual norm gives

$$
\mathcal L_w(u_\theta)
=\frac12\|\mathcal R_\theta\|_{V'}^2,
\qquad
\|\mathcal R_\theta\|_{V'}
=\sup_{v\ne0}\frac{|\mathcal R_\theta(v)|}{\|v\|_V}
=\|e\|_V,
$$

by Cauchy–Schwarz, with equality at $v=e$ when $e\ne0$. Thus

$$
\boxed{\ \mathcal L_w(u_\theta)=\frac12\|\nabla e\|_{L^2}^2=\frac12\|u^\star\|_V^2\,\varepsilon_1^2.\ }
$$

The same Riesz argument applied to $f$ itself gives one more identity we use for normalization later:

$$
\|f\|_{V'}=\sup_{v\ne0}\frac{\int_\Omega fv}{\|v\|_V}=\sup_{v\ne0}\frac{(u^\star,v)_V}{\|v\|_V}=\|u^\star\|_V .
$$

These are identities for the energy norm chosen above. They are not identities for every convention for the $H^{-1}$ norm, every PDE, or every weak loss.

### Does the full weak residual lose information?

For $r\in L^2$, the map $r\mapsto[v\mapsto\int rv]$ is injective because $H_0^1$ is dense in $L^2$. Knowing all weak tests therefore determines $r$, and both full losses vanish only at the solution. But

$$
\|r\|_{V'}\le C_P\|r\|_{L^2},
$$

with no uniform reverse bound on $L^2$. A weaker norm changes *sensitivity* to errors without making nonzero residuals invisible. Actual invisibility enters only when we restrict the tests.

## 2. Conditioning depends on the residual norm

Let $\{\phi_k\}$ be an $L^2$-orthonormal Dirichlet Laplacian eigenbasis, with $-\Delta\phi_k=\lambda_k\phi_k$. For $e=\sum_k e_k\phi_k$,

$$
\mathcal L_s=\frac12\sum_k\lambda_k^2|e_k|^2,
\qquad
\mathcal L_w=\frac12\sum_k\lambda_k|e_k|^2.
$$

The strong loss penalizes high frequencies more heavily. On the first $N$ modes, in the **Euclidean coordinates $(e_1,\ldots,e_N)$**, the Hessians are diagonal:

$$
H_s=\operatorname{diag}(\lambda_k^2),
\qquad H_w=\operatorname{diag}(\lambda_k).
$$

For $\Omega=(0,1)$, $\lambda_k=(k\pi)^2$, giving exactly

$$
\kappa(H_s)=N^4,\qquad \kappa(H_w)=N^2.
$$

This is a conditioning statement in specified coordinates, not an intrinsic condition number of a functional: in energy-orthonormal coordinates the weak Hessian is the identity.

If $A:V\to V'$ denotes the weak Dirichlet Laplacian, it is also the Riesz map for the energy inner product, so

$$
\|\mathcal R\|_{V'}^2
=\langle\mathcal R,A^{-1}\mathcal R\rangle_{V',V}.
$$

The inverse elliptic operator reweights the residual: the energy-dual weak loss acts as **operator preconditioning**.

### What survives neural parameterization?

Both full losses are strictly convex quadratics in the admissible function $u$. Composing with a nonlinear network $u_\theta$ generally gives nonconvex objectives in $\theta$. For a discretized residual vector $r(\theta)$ and a fixed positive-definite weight matrix $M$,

$$
\mathcal L(\theta)=\frac12r(\theta)^\top M r(\theta),
\qquad
\nabla_\theta^2\mathcal L
=J_r^\top M J_r
+\sum_i(Mr)_i\nabla_\theta^2r_i.
$$

The first term is the Gauss–Newton matrix; the second vanishes at zero residual and is small near it when the residual Hessians stay bounded. Changing $M$ can improve conditioning, but the network Jacobian, parameter redundancies, and optimizer also matter. The modal calculation does **not** guarantee a better-conditioned parameter Hessian or faster neural training.

Integration by parts gives one further, purely computational benefit: the weak Poisson loss needs only first spatial derivatives of the network, the strong loss needs second. That reduces differentiation cost. It does not by itself establish smoother parameter losses, lower sampling variance, or better convergence.

## 3. Finite tests: normalization and missing directions

Choose linearly independent tests $v_1,\ldots,v_m\in V$, let $V_m$ be their span, and define

$$
b_i=\mathcal R_\theta(v_i),\qquad G_{ij}=(v_i,v_j)_V.
$$

Writing $v=\sum_i c_i v_i$ gives $\mathcal R_\theta(v)=c^\top b$ and $\|v\|_V^2=c^\top Gc$. Maximizing over $c$ yields

$$
\boxed{
\mathcal L_{w,m}
=\frac12\sup_{v\in V_m\setminus\{0\}}
\frac{|\mathcal R_\theta(v)|^2}{\|v\|_V^2}
=\frac12 b^\top G^{-1}b.
}
$$

In code, solve $Gz=b$ and compute $b^\top z$; an explicit inverse is unnecessary. Our sine basis makes $G$ diagonal.

### Why the Gram matrix matters

For a nonsingular basis change $\widetilde v_j=\sum_i v_iS_{ij}$,

$$
\widetilde b=S^\top b,\qquad \widetilde G=S^\top GS,
\qquad
\widetilde b^\top\widetilde G^{-1}\widetilde b=b^\top G^{-1}b.
$$

The loss depends on the test **space**, not on its basis. The unweighted alternative $\tfrac12 b^\top b$ lacks this invariance except for special basis changes.

For $v_k=\phi_k=\sqrt2\sin(k\pi x)$ we have $b_k=\lambda_ke_k$ and $G_{kk}=\lambda_k$, so

$$
\mathcal L_{w,m}=\frac12\sum_{k\le m}\lambda_k e_k^2,
\qquad
\mathcal L_{\mathrm{raw},m}=\frac12\sum_{k\le m}\lambda_k^2 e_k^2.
$$

Unweighted sine tests therefore retain the strong loss's modal stiffness on the observed modes, despite using only first network derivatives. Rescaling the tests by $1/k$ changes that unweighted stiffness from $k^4$ to $k^2$; Gram weighting already accounts for the rescaling and gives the same objective in either basis.

### What a finite projection cannot see

Let $P_m:V\to V_m$ be the energy-orthogonal projection. For Poisson,

$$
\mathcal L_{w,m}=\frac12\|P_me\|_V^2,
\qquad
\mathcal L_w-\mathcal L_{w,m}
=\frac12\|(I-P_m)e\|_V^2 .
$$

Dividing by $\tfrac12\|u^\star\|_V^2$ makes the consequence concrete, and is exactly how the experiments are normalized:

$$
\underbrace{\frac{b^\top G^{-1}b}{\|f\|_{V'}^2}}_{\text{what training sees}}
=\frac{\|P_me\|_V^2}{\|u^\star\|_V^2},
\qquad
\underbrace{\varepsilon_1^2}_{\text{true error}}
=\frac{\|P_me\|_V^2+\|(I-P_m)e\|_V^2}{\|u^\star\|_V^2}.
$$

**The tested loss is the squared relative energy error, restricted to the test space.** It is bounded by $\varepsilon_1^2$, and everything in the orthogonal complement is free. With the first $m$ sine tests, any error $e=a\phi_K$ with $K>m$ is exactly invisible. For nested dense test spaces the projected losses converge upwards to the full loss **for a fixed error**; this does not guarantee monotone error across separately trained networks.

On a prescribed *linear* trial-error space $E_h$, the gap is controlled by

$$
\beta_{h,m}
=\inf_{e\in E_h\setminus\{0\}}
\frac{\|P_me\|_V}{\|e\|_V},
\qquad\text{giving}\qquad
\|e\|_V\le\frac{\sqrt{2\mathcal L_{w,m}}}{\beta_{h,m}}\ \ \text{when }\beta_{h,m}>0 .
$$

Over all of $V$, any finite test space has $\beta=0$. A finite-parameter neural network is not generally a finite-dimensional linear function space, so such a bound must be established on its attainable error set, or locally on tangent directions; counting parameters and tests is not enough. Finitely sampled strong losses can likewise miss errors between collocation points. Both formulations need independent validation.

## 4. Experiments

### 4.1 A controlled conditioning calculation

Before any network, take the trial space to be the first $N=16$ sine modes and optimize the coefficients $e_k$ directly. The initial error is $e_k^{(0)}=1/k$, chosen so that every mode starts with the *same* energy, $\lambda_k|e_k^{(0)}|^2=\pi^2$: no mode is favored, and the decay of the energy error reflects conditioning alone. Each quadratic runs 2,000 gradient descent steps at its own stable step size $1/\lambda_{\max}(H)$, so

$$
e_k^{(t)}=\left(1-\frac{h_k}{\max_j h_j}\right)^t e_k^{(0)},
\qquad h_k=\lambda_k^2\ \text{(strong) or}\ \lambda_k\ \text{(weak)} .
$$

This is exact and deterministic: no neural parameterization, quadrature, or stochasticity. Because each objective uses its own curvature-based step size, overall loss scaling cannot explain the comparison.

![Modal Hessian growth, gradient descent energy error, and the effect of test-basis rescaling.](conditioning.png)

*Left and right: modal curvature $h_k$ normalized by the first mode, so the curves are $k^4$ and $k^2$. Middle: energy error relative to its initial value, $\|e^{(t)}\|_V/\|e^{(0)}\|_V$ (there is no $u^\star$ here, only an error to contract). The condition numbers at $N=16$ are $16^4=65{,}536$ and $16^2=256$. Mode $k$ contracts by the factor $1-(k/16)^p$ per step, with $p=4$ strong and $p=2$ weak, so the asymptotic rate is set by the slowest mode $k=1$: after 2,000 updates the strong run has cleared all but the two lowest modes yet shrunk mode 1 by only $3\%$, stalling near $0.3$, while the weak run has contracted every mode including $k=1$. The right panel changes only the basis scaling within one fixed test space: the two raw curves differ, while Gram weighting gives the same objective in either basis.*

### 4.2 Neural Poisson benchmark

On $(0,1)$, manufacture the solution and forcing

$$
u^\star(x)=\sin(\pi x)+0.1\sin(8\pi x),
\qquad
f(x)=\pi^2\sin(\pi x)+6.4\pi^2\sin(8\pi x).
$$

The $k=8$ component has small displacement amplitude but carries $0.64/1.64\approx39\%$ of the energy. We use

$$
u_\theta(x)=x(1-x)N_\theta(2x-1),
$$

where $N_\theta$ is a tanh network with two hidden layers of width 32 (1,153 parameters). The boundary condition is exact, so no boundary-penalty weight enters the comparison.

All methods share the same 128-point Gauss–Legendre quadrature, CPU float64, full-batch Adam at learning rate $10^{-3}$, and 3,000 updates. Seeds 0, 1, and 2 are paired across methods: each seed gives identical initial parameters for every loss. We report the final iterate, never a checkpoint selected using the exact solution.

The methods are strong training, Gram-weighted sine tests with $m\in\{2,4,8,16,32\}$, and unweighted sine tests with $m=16$. Dividing out the forcing scale, the implemented objectives are

$$
\widehat{\mathcal L}_s=\frac{\|r\|_{L^2}^2}{\|f\|_{L^2}^2},
\qquad
\widehat{\mathcal L}_{w,m}=\frac{b^\top G^{-1}b}{\|f\|_{V'}^2},
\qquad
\widehat{\mathcal L}_{\mathrm{raw},m}=\frac{b^\top b}{\|f\|_{L^2}^2}.
$$

The weak denominator $\|f\|_{V'}^2=\sum_k\langle f,\phi_k\rangle^2/\lambda_k$ is fixed across all $m$, computed from the forcing's first 32 sine coefficients, which are its complete spectrum here. The first two normalizations turn the losses into squared relative errors, made precise below; the third admits no such reading, which is the point of §3.

On an independent 512-point Gauss–Legendre rule we measure three diagnostics — the energy error $\varepsilon_1$ of §1, plus

$$
\varepsilon_0=\frac{\|u_\theta-u^\star\|_{L^2}}{\|u^\star\|_{L^2}},
\qquad
\rho=\frac{\|-u_\theta''-f\|_{L^2}}{\|f\|_{L^2}} .
$$

By construction $\widehat{\mathcal L}_s=\rho^2$ exactly, and $\widehat{\mathcal L}_{w,m}=\varepsilon_1^2$ restricted to the tested modes, converging to $\varepsilon_1^2$ as $m\to\infty$. Both $\varepsilon_0$ and $\varepsilon_1$ require the exact solution; training uses only the forcing and the tested residuals.

![Strong, weighted weak, and unweighted weak training objectives, with common energy errors versus updates and training time.](training.png)

*Four representative methods; the full sweep over $m$ is in §4.3. Lines show medians over three seeds, shading the full range. The left panel tracks each method's own objective — these are not comparable to each other. The middle and right panels use the common metric $\varepsilon_1$. Time includes optimization steps only, excluding diagnostics, and uses median cumulative training time at each recorded update.*

Final medians over three seeds:

| Objective | Training loss $\widehat{\mathcal L}$ | $L^2$ error $\varepsilon_0$ | Energy error $\varepsilon_1$ | Residual $\rho$ | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong | 6.02e-04 | 0.44% | 0.71% | 2.45% | 7.39 |
| weak-2 | 3.75e-34 | 10.19% | 62.71% | 98.95% | 5.66 |
| weak-4 | 1.69e-21 | 9.95% | 62.47% | 98.81% | 5.30 |
| weak-8 | 3.30e-12 | 15.51% | 118.70% | 276.90% | 5.01 |
| weak-16 | 5.62e-22 | 0.40% | 6.68% | 30.57% | 4.41 |
| weak-32 | 1.09e-04 | 0.06% | 1.27% | 8.82% | 4.49 |
| raw-16 | 1.53e-02 | 18.83% | 53.26% | 194.97% | 4.69 |

The identities above make the first column directly readable. For strong, $\sqrt{6.02\text{e-}4}=2.45\%=\rho$. For weak-32, $\sqrt{1.09\text{e-}4}=1.04\%$ is the energy error *inside* the 32-mode test space, against a true $\varepsilon_1$ of $1.27\%$; the untested remainder is $\sqrt{1.27^2-1.04^2}=0.73\%$. For weak-16, the tested energy error is $2\times10^{-11}$ while the true one is $6.68\%$ — essentially all of the error sits outside the test space.

The strong formulation attains the lowest median energy and strong-residual error. The 32-test weak formulation attains the lowest median $L^2$ error at roughly $60\%$ of the training time. There is no single winner independent of the error norm.

Gram-weighted training also improves markedly over unweighted training at $m=16$ under this shared Adam configuration ($6.68\%$ versus $53.26\%$ energy error). Yet its almost-zero tested loss coexists with that $6.68\%$. **Successfully optimizing the projected objective does not certify the PDE solution.**

### 4.3 Ablation: how many tests are enough?

![Weak test-count ablation showing final tested losses, independent errors, energy-error histories, and final solutions.](projection.png)

*Markers and error bars show medians and full ranges over three seeds. Top left is each run's own tested objective; top right is the three independent diagnostics. Solution curves use the prespecified seed 0, never the best run. All weak runs share the same network, initialization seeds, optimizer, quadrature, and loss normalization.*

With $m=2$ or $4$, the forcing's eighth mode is never tested. The tested losses reach numerical zero while $\varepsilon_1$ stays near $62.5\%$ — exactly the value obtained by simply omitting the true eighth mode:

$$
\varepsilon_0=\frac{0.1}{\sqrt{1.01}}\approx9.95\%,
\qquad
\varepsilon_1=\sqrt{\frac{0.64}{1.64}}\approx62.47\%.
$$

The bottom-right panel confirms this: the weak-4 solution is the smooth first mode alone.

At $m=8$ the tests cover every mode present in the exact solution, yet training is *worse*: $\varepsilon_1=118.70\%$, above the $100\%$ the zero function would achieve. A tested loss of $3\times10^{-12}$ means $\|P_8e\|_V\approx2\times10^{-6}\|u^\star\|_V$, so *all* of that $118.70\%$ lives above the eighth mode; the solution panel shows it as visible high-frequency oscillation, which the tests cannot penalize and the optimizer has no reason to avoid. **Covering the forcing's frequencies is not sufficient:** the tests must also constrain the unwanted frequencies the trial model can generate.

Increasing to 16 and then 32 tests removes most of that unobserved error. Note that the 32-test *training objective* is four orders of magnitude larger than the 16-test one while its solution is an order of magnitude more accurate. A harder-to-minimize loss can be the more informative one, and comparing training losses across test spaces is meaningless.

### 4.4 Limits of this evidence

The verification checks are listed in [Appendix B](#appendix-b-verification-and-reproduction). Under quadrature refinement the three diagnostics move by less than $5\times10^{-14}$ relatively, twelve orders of magnitude below the differences reported in §4.2 and §4.3, so the projection failure is not a quadrature artifact. These checks do not establish stability for arbitrary networks or test spaces. Three seeds on one smooth one-dimensional PDE are an illustration, not a general performance ranking. Learning rates were shared rather than tuned per method, timings reflect this single-threaded CPU implementation, and no neural Hessian spectrum was measured.

## 5. Practical conclusions

A weak formulation changes two things: the derivatives needed to compute a residual, and — depending on the test norm — its spectral weighting. For Poisson, the full energy-dual norm turns squared Laplacian error into gradient error, improving conditioning in a fixed modal parameterization without making neural training convex.

Finite tests then introduce a separate and larger approximation. Gram weighting removes arbitrary basis dependence but cannot recover directions outside the test space, and the tested loss is blind to them by construction. A useful comparison therefore reports the training objective alongside common solution and residual errors, varies the test-space resolution, and checks quadrature independently. In this example, the most misleading conclusion available would have been to declare the smallest training loss the best solution.

## Appendix A: the Dirichlet energy and the Ritz alternative

A third object is often called an energy, and it is worth separating from the norm of §1. With the **Dirichlet energy** $E(u)=\tfrac12\|u\|_V^2-\int_\Omega fu$, expanding about $u^\star$ and using $(u^\star,e)_V=\int fe$ gives

$$
E(u_\theta)-E(u^\star)=\tfrac12\|e\|_V^2=\mathcal L_w(u_\theta).
$$

So for this symmetric problem the full weak loss *is* the excess Dirichlet energy, and minimizing $E$ directly — the [Deep Ritz](https://arxiv.org/abs/1710.00211) approach — needs no test functions at all. We exclude it from the comparison deliberately: it has no notion of an untested direction, since $E$ sees all of $e$, so it cannot exhibit the failure mode §4.3 is built to expose. The identity depends on the operator being symmetric. For nonsymmetric problems such as advection–diffusion no such energy functional exists, while the weak residual formulation of §1 still applies.

## Appendix B: verification and reproduction

Every run is checked twice against a refined quadrature. Doubling the diagnostic rule from 512 to 1,024 Gauss–Legendre points changes $\varepsilon_0$, $\varepsilon_1$, and $\rho$ by less than $5\times10^{-14}$ relatively. Doubling the training rule from 128 to 256 points changes the normalized objective by less than $2.2\times10^{-8}$ absolutely; that worst case is raw-16, whose objective is $1.5\times10^{-2}$, and each weak run's own gap stays far below its own reported loss. The script additionally asserts, to $10^{-9}$ or tighter:

- the manufactured pair satisfies $-u^{\star\prime\prime}=f$;
- integration by parts holds discretely on the quadrature rule, so $b$ computed from $u'$ matches $b$ computed from $-u''$;
- the sine Gram matrix equals $\operatorname{diag}(\lambda_k)$, and $b^\top G^{-1}b$ is invariant under a random nonorthogonal change of basis;
- the analytically invisible mode $e=0.1\sin(8\pi x)$ gives $b_{1:4}=0$ while $b^\top G^{-1}b$ equals its true energy;
- parameter gradients of all three training objectives match central finite differences, here to a relative $10^{-5}$ set by the accuracy of the difference itself.

To reproduce, from the repository root with NumPy, Matplotlib, and CPU PyTorch:

```bash
python code/strong-weak-pinns/experiment.py              # 21 runs, checks, figures
python code/strong-weak-pinns/experiment.py --plot-only  # figures from committed measurements
```

The [experiment README](https://github.com/dani2442/dani2442_code/blob/main/strong-weak-pinns/README.md) covers environment setup, output formats, and options for separate runs.

## References

- Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://doi.org/10.1016/j.jcp.2018.10.045). *Journal of Computational Physics*, 378, 686–707.
- E, W. and Yu, B. (2018). [The Deep Ritz Method: A Deep Learning-Based Numerical Algorithm for Solving Variational Problems](https://arxiv.org/abs/1710.00211). *Communications in Mathematics and Statistics*, 6, 1–12. Discussed in Appendix A as the energy-minimization alternative excluded here.
- Kharazmi, E., Zhang, Z., and Karniadakis, G. E. (2019). [Variational Physics-Informed Neural Networks for Solving Partial Differential Equations](https://arxiv.org/abs/1912.00873). The variational residual construction motivates the finite-test comparison; the Poisson identities and controlled ablations above are derived explicitly here.
