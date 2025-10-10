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

## Kalman controllability criterion
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

Direction 1 (rank condition fails => not controllable)

If $\operatorname{rank}\mathcal{C}<n$ then there exists a nonzero vector $q\in\mathbb{R}^n$ such that
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

---

## If $\operatorname{rank}(\mathcal{C}) = n$ then controllable (Gramian construction)

Define the finite-horizon controllability Gramian for $T>0$:
$$
W_c(T) \;=\; \int_0^T e^{A\tau} B B^T e^{A^T\tau}\, d\tau.
$$
Two facts are central:

1. For any desired displacement $d\in\mathbb{R}^n$ there exists an input $u(\cdot)$ steering $x(0)=0$ to $x(T)=d$ if and only if $d$ lies in the column space (image) of $W_c(T)$. In particular, if $W_c(T)$ is invertible (positive definite) then every $d$ is reachable at time $T$.

2. $W_c(T)$ is positive definite for some (equivalently, sufficiently large) $T>0$ if and only if $\operatorname{rank}\mathcal{C}=n$.

We outline why these hold and how to construct a minimum-energy control.

---

### Why the Gramian gives reachability and an explicit minimum-energy control

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

---

### Why invertibility of $W_c(T)$ is equivalent to the Kalman rank condition

If $\operatorname{rank}\mathcal{C}<n$ then, as shown earlier, there exists $q\neq0$ with $q^T A^k B=0$ for $k=0,\dots,n-1$. This implies $B^T e^{A^T\tau}q\equiv0$ and hence
$$
q^T W_c(T) q = \int_0^T \|B^T e^{A^T\tau}q\|^2\,d\tau = 0,
$$
so $W_c(T)$ is singular for every $T>0$.

Conversely, if $\operatorname{rank}\mathcal{C}=n$ but $W_c(T)$ were singular for every $T$, there would exist $q\neq0$ with $q^T W_c(T) q=0$ for all $T$. Hence $B^T e^{A^T\tau} q\equiv0$ for all $\tau\ge0$, and differentiating at $\tau=0$ repeatedly yields
$$
B^T (A^T)^k q = 0\qquad\text{for all }k\ge0,
$$
equivalently $q^T A^k B=0$ for all $k\ge0$. By Cayley–Hamilton only the first $n$ powers are independent, so this contradicts $\operatorname{rank}\mathcal{C}=n$. Therefore for some $T>0$ the Gramian $W_c(T)$ is invertible, and reachability follows from the construction in the previous section.

---

## Conclusion and explicit control law

Putting the arguments together:

- If $\operatorname{rank}\mathcal{C}<n$ there is a nonzero direction that no input can influence; the system is not controllable.
- If $\operatorname{rank}\mathcal{C}=n$, then for some $T>0$ the Gramian $W_c(T)$ is invertible and the explicit minimum-energy control
$$
u(s)=B^T e^{A^T(T-s)} W_c(T)^{-1} \big(x_{\mathrm{target}}-e^{AT}x_0\big)
$$
steers $x(0)=x_0$ to $x(T)=x_{\mathrm{target}}$. Therefore the Kalman rank condition is equivalent to controllability for finite-dimensional LTI systems.

---

### Intuition

The controllability matrix $\mathcal{C}=[B\; AB\; A^2B\;\dots]$ collects the directions in state space that can be injected through $B$ and propagated by the dynamics $A$. If those propagated directions span $\mathbb{R}^n$ then, by combining time-varying inputs, one can synthesize a control that reaches any target state. If they do not span the state space there exists a direction orthogonal to all those columns that cannot be influenced by any input.

The controllability matrix $\mathcal{C}=[B\; AB\; A^2B\;\dots]$ collects the directions in state space that can be injected through $B$ and propagated by the dynamics $A$. If those propagated directions span $\mathbb{R}^n$ then, by combining time-varying inputs, one can synthesize a control that reaches any target state. If they do not span the state space there exists a direction orthogonal to all those columns that cannot be influenced by any input.

