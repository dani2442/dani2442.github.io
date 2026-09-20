---
title: "Feynman–Kac formula"
date: 2026-09-19
tags: ["stochastic processes", "brownian motion", "monte carlo", "pde"]
categories: ["pde", "probability"]
author: "Daniel López Montero"
showToc: true
draft: false
description: "The Feynman–Kac formula connects stochastic processes to partial differential equations."
ShowWordCount: false
ShowReadingTime: true
comments: true
TocOpen: true
UseHugoToc: true
editPost:
    URL: "https://github.com/dani2442/dani2442.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---



There is a beautiful connection between probability theory and partial differential equations (PDEs), given by the famous Feynman–Kac theorem.

This result allows us to translate between finite-dimensional stochastic problems and infinite-dimensional deterministic problems. It appears throughout finance, physics, control theory, and machine learning (generative modeling and reinforcement learning).

What makes this connection especially useful is that it works in both directions.

The solution to a high-dimensional PDE (which can be very expensive to compute numerically) can be evaluated at a single point simply by simulating a random process. Conversely, a difficult question about a stochastic process can be transformed into a deterministic PDE and tackled using standard techniques.




> **Example.** Take the Dirichlet problem
> $$ \begin{cases} \Delta u = 0 & \text{in } \Omega, \\ u = g & \text{on } \partial\Omega. \end{cases} $$
> Then its solution satisfies
> $$ u(x) \;=\; \mathbb{E}_x\big[\,g(B_\tau)\,\big] $$
> where $B$ is a Brownian motion started at $x$ and $\tau$ is the first time it
> leaves $\Omega$.

![One Brownian path started at x, wandering through the bean-shaped domain until it leaves at the exit point B_tau](fk_bean_walk.gif)


The proof is short and worth seeing once before the general statement. It needs
one tool: the chain rule for Brownian paths.

> **Itô's formula.** For $\varphi\in C^2(\mathbb{R}^d)$ and a Brownian motion $B$,
> $$ d\varphi(B_t) \;=\; \nabla\varphi(B_t)^\top dB_t \;+\; \tfrac12\Delta\varphi(B_t)\,dt . $$
> The second term is what an ordinary chain rule misses. The path is nowhere
> differentiable, but it accumulates quadratic variation at a definite rate,
> $d\langle B^i,B^j\rangle_t=\delta_{ij}\,dt$, and the Laplacian is what that
> second-order term collects.

*Proof of the example.* Let $\Omega$ be bounded, say $\Omega\subset B(0,R)$, let
$u\in C^2(\Omega)\cap C(\overline\Omega)$ solve the problem, let $B$ start at
$x\in\Omega$, and let $\tau=\inf\{s\ge0:\;B_s\notin\Omega\}$. That such a $u$
exists is a separate question; here we take one as given and show it has to be
the average of its own boundary data.

Check first that the path leaves at all. Itô's formula on $h(y)=|y|^2$, for which
$\tfrac12\Delta h=d$, gives
$$
|B_{t\wedge\tau}|^2
=|x|^2+2\int_0^{t\wedge\tau}B_s^\top dB_s+d\cdot(t\wedge\tau),
$$
where $t\wedge\tau=\min(t,\tau)$ means that we stop the path when it exits.
The stochastic integral has expectation zero: before $\tau$, its integrand
stays in the bounded domain, so it is square-integrable on every finite time
interval. Since the stopped path stays in $\overline\Omega$, taking expectations
yields
$$
d\cdot\mathbb{E}_x[t\wedge\tau]
=\mathbb{E}_x\big[|B_{t\wedge\tau}|^2\big]-|x|^2\;\le\;R^2 .
$$
Letting $t\to\infty$ and using monotone convergence, we obtain $\mathbb{E}_x[\tau]\le R^2/d$;
in particular, $\tau<\infty$ almost surely, and since Brownian paths are
continuous, the exit point $B_\tau$ lies on $\partial\Omega$.

Now apply Itô's formula to $u$. Since $\nabla u$ need not stay bounded near
$\partial\Omega$, stop just short of it, at
$$
\tau_n:=\inf\{s\ge0:\;\operatorname{dist}(B_s,\partial\Omega)\le 1/n\},
$$
with $n$ large enough that $\operatorname{dist}(x,\partial\Omega)>1/n$; the
stopped path then lives in a compact subset of $\Omega$, and $\tau_n\uparrow\tau$.
Itô's formula gives
$$
u(B_{t\wedge\tau_n})
=u(x)+\int_0^{t\wedge\tau_n}\nabla u(B_s)^\top dB_s
  +\frac12\int_0^{t\wedge\tau_n}\Delta u(B_s)\,ds.
$$
The last integral vanishes because $u$ is harmonic, and the stochastic integral
has expectation zero because $\nabla u$ is bounded there, so
$\mathbb{E}_x[u(B_{t\wedge\tau_n})]=u(x)$. Since $u$ is continuous on the compact
set $\overline\Omega$, it is bounded, so bounded convergence — first as
$n\to\infty$, then as $t\to\infty$, using that the path does exit and that $u$ is
continuous up to the boundary — turns this into
$$
u(x) \;=\; \mathbb{E}_x\big[\,g(B_\tau)\,\big]. \qquad\blacksquare
$$


## 1. The Feynman–Kac theorem

With the previous example in mind, we now turn to the classical Feynman–Kac
theorem, which connects stochastic differential equations (SDEs) with parabolic
PDEs. The idea is the same: follow a random path and average the data it reaches.
Here we run the process to a fixed terminal time, allowing for drift, a
space-dependent diffusion, discounting, and a source accumulated along the path.

Let $X$ solve the SDE
$$
dX_t \;=\; b(X_t)\,dt + \sigma(X_t)\,dW_t
$$
in $\mathbb{R}^d$, with $b,\sigma$ globally Lipschitz. Its *generator* is the
spatial differential operator
$$
\mathcal{L}\varphi \;:=\; \tfrac12\operatorname{tr}\big(\sigma\sigma^\top D^2\varphi\big)+b\cdot\nabla\varphi .
$$
It is the drift term in Itô's formula for $\varphi(X_t)$: the first-order part
comes from the drift of $X$, and the second-order part comes from its quadratic variation.

> **Theorem (Feynman–Kac).**
> Let $V,g:\mathbb{R}^d\to\mathbb{R}$ and $f:[0,T]\times\mathbb{R}^d\to\mathbb{R}$
> be continuous, with $V\ge0$, and suppose
> $u\in C^{1,2}\big([0,T)\times\mathbb{R}^d\big)\cap C\big([0,T]\times\mathbb{R}^d\big)$ solves
> $$ \partial_t u + \mathcal{L}u - Vu + f \;=\; 0, \qquad u(T,\cdot)=g . $$
> If $u$, $f$, and $\sigma^\top\nabla u$ grow at most polynomially in $x$, uniformly in $t$, then
> $$ u(t,x) \;=\; \mathbb{E}_{t,x}\Big[\, e^{-\int_t^T V(X_r)dr}\,g(X_T) \;+\; \int_t^T e^{-\int_t^s V(X_r)dr}\, f(s,X_s)\,ds \,\Big], $$
> where $\mathbb{E}_{t,x}$ is the expectation for the path started at $X_t=x$.


The [proof in the appendix](#appendix-proof-of-the-feynman-kac-theorem) follows
the same strategy as the Dirichlet example: Itô's formula produces a martingale,
and taking expectations gives the representation.

This statement gives uniqueness for free, but it presumes that a solution exists.
The converse of the theorem (that the expectation on the right-hand side is itself a solution of
the PDE) needs considerably more technical work. That construction is carried out in Friedman [5], Ch. 6, §§4–5.

> **Historical note.** Feynman (1948) described the evolution of the Schrödinger
> equation
> $$ i\hbar\,\partial_t\psi \;=\; -\tfrac{\hbar^2}{2m}\Delta\psi + V\psi $$
> by summing over *every* path joining the two endpoints, each weighted by a
> complex number of modulus one.
>
> Kac (1949) later developed a related idea for the heat equation
> (the Schrödinger equation in imaginary time):
> $$ \partial_t u \;=\; \tfrac12\Delta u - Vu . $$
> He observed that the solution can be written as an average over random Brownian paths:
> $$ u(t,x) \;=\; \mathbb{E}_x\Big[e^{-\int_0^t V(B_s)ds}\,g(B_t)\Big]. $$

Black–Scholes (1973) is the same theorem read through the finance dictionary.
Take the risk-neutral geometric Brownian motion $dX_t=rX_t\,dt+\sigma X_t\,dW_t$
with $V\equiv r$ constant, $f=0$, and $g$ the payoff at maturity: the PDE is
$$
\partial_t u + \tfrac12\sigma^2x^2\,\partial_{xx}u + rx\,\partial_x u - ru = 0,
\qquad u(T,\cdot)=g,
$$
and the representation is the pricing formula
$u(t,x)=e^{-r(T-t)}\,\mathbb{E}\big[g(X_T)\mid X_t=x\big]$. Every ingredient has
a name on both sides: $V$ is the discount rate, a source $f$ is a dividend or
running payoff, and $\sigma^\top\nabla u$ is the hedging portfolio. Read in that
dictionary, the proof in the appendix is the statement that a hedged position has
no drift.


## 2. Feynman–Kac family of equations

Look back at the proof and notice how little of it was about the heat equation.
Itô's formula turned the operator into a drift, the equation cancelled that
drift, and what was left was a martingale. Not a single step of that argument asks
which operator it was handed. Change the operator and the same three lines still
run.

The dictionary below collects the correspondences
in that order, with $\mathcal{L}$ the generator of $dX_t=b(X_t)\,dt+\sigma(X_t)\,dW_t$ as above
and, for a path started at time $t_0$, the exit time defined by
$$
\tau=\inf\{t>t_0:\,X_t\notin\Omega\}.
$$

![Dictionary of Feynman–Kac correspondences: linear and semilinear parabolic and elliptic equations, Fokker–Planck, the principal eigenvalue problem and HJB, each with its stochastic object and its representation](feynman_kac-feynman-kac.png)

### 2.1 Boundary conditions

| Name | PDE ingredient | Stochastic object |
|---|---|---|
| Dirichlet condition | $u=g$ on $\partial\Omega$ | process killed on contact |
| Neumann condition | $\partial_n u=0$ | reflected diffusion |
| Robin condition | $\partial_n u=\alpha u$ | reflected diffusion killed at rate $\alpha$ in local time |


### 2.2 Other families

So far, we have only considered stochastic processes driven by Brownian motion, but the Feynman–Kac theorem is more general.

#### A finite state space

Let $\xi$ be a continuous-time Markov
chain on $\{1,\dots,n\}$ with generator matrix $Q$, and let $V\in\mathbb{R}^n$ be a killing rate.
The "PDE" is then a linear system of ODEs,
$$
u'(t) \;=\; \big(Q-\operatorname{diag}V\big)\,u(t),
\qquad u(0)=g\in\mathbb{R}^n ,
$$
and the Feynman–Kac formula reads
$$
u_i(t)\;=\;\mathbb{E}_i\Big[e^{-\int_0^t V(\xi_s)\,ds}\,g(\xi_t)\Big],
\qquad\text{that is}\qquad u(t)=e^{t(Q-\operatorname{diag}V)}\,g .
$$
The generator has replaced the Laplacian and a matrix exponential has replaced
the heat semigroup; not a single other word of the argument changes. It is a
useful version to keep in mind because it makes plain that the theorem is a
statement about generators and semigroups rather than Brownian motion specifically.

#### Lévy processes

Replace $W$ by a Lévy process. The simplest case is a compound Poisson path
that waits for an exponentially distributed time with rate $\lambda$ and then jumps,
$$
X_t \;=\; x+\sum_{i=1}^{N_t}Z_i ,
$$
with $N_t$ a Poisson process of rate $\lambda$ and the $Z_i$ independent draws
from a distribution $\mu$. Its generator is an integral operator,
and the equation becomes
$$
\begin{cases}
\partial_t u(t,x) = \lambda\!\int\big[u(t,x+z)-u(t,x)\big]\,\mu(dz)\;-\;V(x)\,u(t,x),\\
 u(0,\cdot)=g,
\end{cases}
$$
while the representation remains unchanged:
$$ u(t,x)=\mathbb{E}_x\big[e^{-\int_0^tV(X_s)ds}g(X_t)\big].
$$
What is new is that the operator is *nonlocal*: the
value at $x$ is tied to the value at every $x+z$ the path can reach in one jump,
not merely to an infinitesimal neighborhood.

Letting the jump measure be
$\nu(dz)=c_{d,\alpha}|z|^{-d-\alpha}dz$ makes $X$ an $\alpha$-stable process and
$\mathcal{L}=-(-\Delta)^{\alpha/2}$, so the fractional heat equation is
Feynman–Kac for a pure-jump path. Nonlocality does cost something at the
boundary: a jump can leave $\Omega$ without ever touching $\partial\Omega$, so a
Dirichlet datum must be prescribed on the whole complement $\Omega^c$.


#### Deterministic flows

Take the opposite extreme and switch the noise off, $\sigma\equiv0$. The process
is the ODE flow $\dot X_s=b(X_s)$ started at $X_t=x$; its law is a Dirac mass,
and the expectation in the theorem has nothing left to average over:
$$
u(t,x)\;=\;e^{-\int_t^TV(X_r)dr}\,g(X_T)\;+\;\int_t^Te^{-\int_t^sV(X_r)dr}\,f(s,X_s)\,ds .
$$
This is the method of characteristics for the linear transport equation
$\partial_tu+b\cdot\nabla u-Vu+f=0$: the characteristic through $(t,x)$ is the
trajectory itself, $V$ damps the datum carried along it, $f$ feeds it, and with
$V=f=0$ the solution is the terminal value transported back, $u(t,x)=g(X_T)$.

Two things do change, and they are what make the degenerate case worth stating
rather than dismissing.

1. **The boundary.** A Brownian path hits every point of $\partial\Omega$, which
   is why the Dirichlet problem of the opening example takes data on all of it; a
   trajectory arrives at exactly one point, so data may be prescribed only where
   characteristics enter, on the inflow boundary
   $\{y\in\partial\Omega:\,b(y)\cdot n(y)<0\}$, and prescribing it anywhere else
   overdetermines the problem. That is the degenerate end of the dictionary of
   §2.1, and the mirror image of the nonlocal case, where a jump can leave
   $\Omega$ without ever touching $\partial\Omega$ and the datum has to be given
   on the whole complement.

2. **Regularity.** Diffusion smooths; transport does not. With
   $\sigma\sigma^\top$ nondegenerate the representation is an average over
   infinitely many paths and $u$ is smooth however rough $g$ is, whereas here
   $u(t,\cdot)$ is exactly $g$ transported, and a discontinuity in the datum
   travels along its characteristic forever.



## References

[1] R. P. Feynman. Space-Time Approach to Non-Relativistic Quantum Mechanics. *Reviews of Modern Physics* **20** (1948) 367–387.

[2] M. Kac. On Distributions of Certain Wiener Functionals. *Transactions of the American Mathematical Society* **65** (1949) 1–13.

[3] F. Black and M. Scholes. The Pricing of Options and Corporate Liabilities. *Journal of Political Economy* **81** (1973) 637–654.

[4] I. Karatzas and S. Shreve. *Brownian Motion and Stochastic Calculus*. 2nd ed., Springer, 1991. (§4.2–4.4 for the Dirichlet problem, §5.7 for Feynman–Kac.)

[5] A. Friedman. *Stochastic Differential Equations and Applications*, Vol. 1. Academic Press, 1975. (Ch. 6, §§4–5: fundamental solutions for parabolic equations and the stochastic representation of solutions.)


## Appendix: Proof of the Feynman–Kac theorem {#appendix-proof-of-the-feynman-kac-theorem}

Fix $t<T$ and start the SDE at $X_t=x$. The discount factor
$$
D_s:=e^{-\int_t^s V(X_r)\,dr}
$$
satisfies $D_t=1$ and $dD_s=-V(X_s)D_s\,ds$. Since $V\ge0$, we also have
$0<D_s\le1$.

First apply Itô's formula to the time-dependent function $u(s,X_s)$, for $s<T$:
$$
du(s,X_s)
=\big(\partial_s u+\mathcal{L}u\big)(s,X_s)\,ds
 +\nabla u(s,X_s)^\top\sigma(X_s)\,dW_s.
$$
The product rule gives $d(D_su)=D_s\,du+u\,dD_s$; there is no quadratic
covariation term because $D$ has finite variation. Consequently,
$$
d\big(D_su(s,X_s)\big)
=D_s\big(\partial_s u+\mathcal{L}u-Vu\big)(s,X_s)\,ds
 +D_s\nabla u(s,X_s)^\top\sigma(X_s)\,dW_s.
$$
The PDE says that the drift in parentheses is $-f$. Adding the accumulated
source therefore cancels it: if
$$
Y_s:=D_su(s,X_s)+\int_t^s D_r f(r,X_r)\,dr,
$$
then
$$
Y_s=u(t,x)+\int_t^s D_r\nabla u(r,X_r)^\top\sigma(X_r)\,dW_r.
$$

To take expectations, we must check that this stochastic integral is a true
martingale. Globally Lipschitz coefficients give finite moments on a finite
time interval:
$$
\mathbb{E}_{t,x}\!\left[\sup_{t\le r\le T}|X_r|^p\right]<\infty
\qquad\text{for every }p\ge2.
$$
By the polynomial growth assumption, choose $C>0$ and $m\ge1$ such that
$|\sigma^\top\nabla u(r,y)|\le C(1+|y|^m)$. Together with $D_r\le1$, this gives
$$
\mathbb{E}_{t,x}\!\left[\int_t^T
 D_r^2\,|\sigma(X_r)^\top\nabla u(r,X_r)|^2\,dr\right]
\le 2C^2(T-t)\,\mathbb{E}_{t,x}\!\left[1+\sup_{t\le r\le T}|X_r|^{2m}\right]
<\infty.
$$
Thus the stochastic integral is square-integrable and has expectation zero.
For every $s<T$ we obtain
$$
u(t,x)=\mathbb{E}_{t,x}\!\left[
D_su(s,X_s)+\int_t^s D_r f(r,X_r)\,dr\right].
$$

Finally, let $s\uparrow T$. Continuity and the terminal condition give
$u(s,X_s)\to g(X_T)$ almost surely. The polynomial growth of $u$ and $f$
bounds the expression inside the expectation by
$C'(1+\sup_{t\le r\le T}|X_r|^q)$ for suitable $C'>0$ and $q\ge2$.
This bound is integrable by the same moment estimate, so dominated convergence
yields
$$
u(t,x)=\mathbb{E}_{t,x}\!\left[
e^{-\int_t^T V(X_r)\,dr}\,g(X_T)
+\int_t^T e^{-\int_t^s V(X_r)\,dr}\,f(s,X_s)\,ds
\right]. \qquad\blacksquare
$$
