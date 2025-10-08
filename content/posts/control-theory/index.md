---
title: "Control Theory"
date: 2025-10-03
tags: ["machine learning", "gaussian processes", "probability", "stochastic processes", "kernel methods"]
author: "Daniel López Montero"
showToc: true
draft: true
description: "Math and Code"
ShowWordCount: false
ShowReadingTime: true
comments: true
UseHugoToc: true
editPost:
    URL: "https://github.com/dani2442/dani2442.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---








# Statement

Consider the linear time-invariant system
[
\dot x(t)=Ax(t)+Bu(t),\qquad x\in\mathbb{R}^n,;u\in\mathbb{R}^m,
]
with constant matrices (A\in\mathbb{R}^{n\times n}), (B\in\mathbb{R}^{n\times m}).
**Kalman (controllability) criterion:** the pair ((A,B)) is (completely) controllable iff the (n\times nm) **controllability matrix**
[
\mathcal{C} ;=; \big[,B;|;AB;|;A^2B;|;\dots;|;A^{n-1}B,\big]
]
has rank (n).

Below I give a step-by-step derivation / proof of this equivalence and show how to construct a control input when the criterion holds.

---

# 1 — preliminaries: solution and reachable set

The solution with initial condition (x(0)=x_0) is
[
x(T)=e^{AT}x_0 + \int_{0}^{T} e^{A(T-s)}B,u(s),ds.
]
Define the **reachable set from (x_0) at time (T)**:
[
\mathcal{R}_T(x_0);=;\Big{,x(T);\Big|;x(T)=e^{AT}x_0+\int_0^T e^{A(T-s)}B u(s),ds,;u(\cdot)\ \text{admissible}\Big}.
]
From the controllability point of view it suffices to study reachability from (x_0=0); write
[
\mathcal{R}_T;=;\Big{,\int_0^T e^{A(T-s)}B u(s),ds ;\Big|; u(\cdot)\Big}.
]
The system is (completely) controllable iff for some (equivalently every) (T>0) we have (\mathcal{R}*T=\mathbb{R}^n) (or equivalently (\bigcup*{T>0}\mathcal{R}_T=\mathbb{R}^n)).

---

# 2 — direction “if rank (\mathcal{C}<n), then not controllable” (easy/standard)

Assume (\operatorname{rank}\mathcal{C}<n). Then the rows of (\mathcal{C}) have a nontrivial left null vector: there exists (q\in\mathbb{R}^n), (q\neq 0), such that
[
q^T A^k B = 0\qquad\text{for }k=0,1,\dots,n-1.
]
But (e^{A\tau} = \sum_{k=0}^\infty \dfrac{\tau^k}{k!} A^k). Therefore for every (\tau\ge0)
[
q^T e^{A\tau} B
= \sum_{k=0}^\infty \frac{\tau^k}{k!}, q^T A^k B
= 0,
]
(the infinite series has all coefficients zero because the first (n) moments are zero and Cayley–Hamilton shows higher powers are linear combinations of the first (n)). Hence
[
q^T!\left(\int_0^T e^{A(T-s)}B,u(s),ds\right)=\int_0^T q^T e^{A(T-s)}B,u(s),ds = 0
]
for every admissible (u). So for any initial (x_0),
[
q^T x(T) = q^T e^{AT}x_0
]
is independent of the choice of (u). Therefore one cannot steer the component (q^T x) arbitrarily; the reachable set cannot be all of (\mathbb{R}^n). So the system is **not controllable**.

This proves the contrapositive: if the system is controllable then (\operatorname{rank}\mathcal{C}=n).

---

# 3 — direction “if (\operatorname{rank}\mathcal{C}=n) then controllable” (constructive via the controllability Gramian)

Define the (finite-horizon) controllability Gramian for (T>0):
[
W_c(T);=;\int_0^T e^{A\tau} B B^T e^{A^T\tau},d\tau.
]
Two facts we use:

**(a)** For any (d\in\mathbb{R}^n) there exists an input (u(\cdot)) steering (x(0)=0) to (x(T)=d) iff (d) lies in the column space (image) of (W_c(T)). In particular if (W_c(T)) is invertible (i.e. positive definite) then every (d) is reachable at time (T).

**(b)** (W_c(T)) is positive definite for some (equivalently every sufficiently small) (T>0) if and only if (\operatorname{rank}\mathcal{C}=n).

I now justify these two claims.

---

### 3.1 Why (W_c(T)) gives reachability and yields an explicit control (minimum-energy)

Given a desired displacement (d = x(T)-e^{AT}x_0), we want (u(\cdot)) so that
[
\int_0^T e^{A(T-s)}B,u(s),ds = d.
]
Among all controls that achieve this, the minimum-energy control (minimizes (J(u)=\tfrac12\int_0^T |u(s)|^2 ds)) can be found by Lagrange multipliers: define Lagrangian
[
\mathcal{L}(u,\lambda)=\tfrac12\int_0^T u^T u,ds + \lambda^T\Big(\int_0^T e^{A(T-s)}B u(s),ds - d\Big).
]
Stationarity w.r.t. (u) gives (pointwise)
[
u(s) + B^T e^{A^T(T-s)}\lambda = 0 \quad\Rightarrow\quad u(s) = -B^T e^{A^T(T-s)}\lambda.
]
Plugging into the constraint,
[
d = \int_0^T e^{A(T-s)}B,u(s),ds
= -\int_0^T e^{A(T-s)}B,B^T e^{A^T(T-s)}\lambda,ds
= -W_c(T),\lambda.
]
Hence (\lambda = -W_c(T)^{-1} d) provided (W_c(T)) is invertible, and the minimum-energy control is
[
\boxed{ ;u^\star(s) ;=; B^T e^{A^T(T-s)} W_c(T)^{-1} d ; }.
]
Substituting back shows that this (u^\star) indeed produces (x(T)=e^{AT}x_0+d=x_{\text{target}}). So invertibility of (W_c(T)) implies full reachability at time (T).

---

### 3.2 Why invertibility of (W_c(T)) is equivalent to Kalman rank condition

Suppose (\operatorname{rank}\mathcal{C}<n). Then, as in section 2, there exists (q\ne 0) with (q^T A^k B = 0) for all (k=0,\dots,n-1). Then for all (\tau\ge0),
[
B^T e^{A^T\tau} q= \sum_{k=0}^\infty \frac{\tau^k}{k!} B^T A^{T k} q = 0,
]
so (B^T e^{A^T\tau} q\equiv 0) on ([0,T]). Therefore
[
q^T W_c(T) q = \int_0^T |B^T e^{A^T\tau} q|^2 d\tau =0,
]
so (W_c(T)) is singular. Thus (\operatorname{rank}\mathcal{C}<n \Rightarrow W_c(T)) singular for every (T).

Conversely, suppose (\operatorname{rank}\mathcal{C}=n). If (W_c(T)) were singular for every (T>0), there would exist (q\neq 0) with (q^T W_c(T) q =0) for all (T). But (q^T W_c(T) q = \int_0^T |B^T e^{A^T\tau} q|^2 d\tau), so this forces (B^T e^{A^T\tau} q\equiv 0) on ([0,T]) for every (T), hence on ([0,\infty)). Differentiating at (\tau=0) repeatedly shows (B^T A^{T k} q = 0) for all (k\ge0), or equivalently (q^T A^k B = 0) for all (k\ge0). That contradicts (\operatorname{rank}\mathcal{C}=n) (because Cayley–Hamilton implies only the first (n) powers are independent). So there must exist some (T>0) for which (W_c(T)) is invertible. By 3.1 this implies reachability to any (x(T)). Hence (\operatorname{rank}\mathcal{C}=n\Rightarrow) controllable.

---

# 4 — conclusion and explicit control law

Putting the two directions together:

* If (\operatorname{rank}\mathcal{C}<n) there exists a direction that cannot be influenced by any input, so the system is not controllable.
* If (\operatorname{rank}\mathcal{C}=n), then for some (T>0) the Gramian (W_c(T)) is invertible and the explicit minimum-energy control
  [
  \boxed{ ;u(s)=B^T e^{A^T(T-s)} W_c(T)^{-1} \big(x_{\text{target}}-e^{AT}x_0\big); }
  ]
  steers (x(0)=x_0) to (x(T)=x_{\text{target}}). Therefore the system is controllable.

Thus the Kalman rank condition is equivalent to controllability for linear time-invariant finite-dimensional systems.

---

# Short remark (intuition)

The controllability matrix (\mathcal{C}=[B; AB; A^2B;\dots]) collects the directions you can instantaneously inject into the state ((B)) and how the dynamics (A) propagate those directions (the columns (AB,A^2B,\dots)). If those propagated directions span the whole state space, you can synthesize inputs (e.g. via the Gramian construction) to reach any state. If they don't, there is a leftover direction orthogonal to all those columns you simply *cannot* influence.

---

If you want, I can (1) show a small (2\times2) numerical example and compute the Gramian and the explicit control, or (2) show the equivalent Hautus test version (\operatorname{rank}[,\lambda I - A;,;B,]=n\ \forall \lambda\in\mathbb{C}) and how it relates. Which would you prefer?
