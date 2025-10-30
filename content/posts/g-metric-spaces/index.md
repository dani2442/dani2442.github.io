---
title: "G-Metric Spaces"
date: 2025-10-30
tags: ["machine learning", "analysis", "kernel methods"]
categories: ["analysis", "machine learning"]
author: "Daniel López Montero"
showToc: true
draft: false
description: "A generalization of metric spaces."
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

It is often useful to consider distances on triples of points rather than only on pairs. This motivates the notion of G-metric spaces, which extend the usual (pairwise) metric to a symmetric, three-argument distance.

A metric space is a pair $(X,d)$ where $X$ is a set and $d: X \times X \to \mathbb{R}_{\ge 0}$ satisfies:
1. $d(x,x)=0$.
2. $d(x,y) > 0$ if $x\neq y$ (positivity/identity of indiscernibles).
3. $d(x,y)=d(y,x)$ (symmetry).
4. $d(x,z)\le d(x,y)+d(y,z)$ (triangle inequality).

This structure underlies the standard notions of convergence, continuity, compactness, and completeness.

## 1. G-Metric Spaces

Following Mustafa–Sims, a G-metric on a set $X$ is a map $G: X \times X \times X\to \mathbb{R}_{\ge 0}$ such that, for all $x,y,z,a\in X$:
1. $G(x,y,z)=0$ if and only if $x=y=z$.
2. $G(x,x,y)>0$ whenever $x\neq y$.
3. $G(x,x,y)\le G(x,y,z)$ whenever $y\neq z$.
4. $G$ is symmetric in its three arguments.
5. $G(x,y,z)\le G(x,a,a)+G(a,y,z)$.

When these axioms hold, $(X,G)$ is called a G-metric space. Two simple examples (built from an ordinary metric $d$) are
$$
G(x,y,z)=\max\{d(x,y),\,d(y,z),\,d(z,x)\},\qquad
G(x,y,z)=d(x,y)+d(y,z)+d(z,x).
$$
Both satisfy the axioms above whenever $d$ is a metric. The concept naturally extends to $n$-ary (higher-order) distances, but we focus on the triadic case for clarity.

## 2. Topology, Convergence and Completeness

The G-metric generates a natural topology on $X$ via G-balls. For $x_0\in X$ and $r>0$, define the $G$-ball
$$
B_G(x_0, r) := \{ y\in X : G(x_0, y, y)\lt r \}.
$$
One checks that it is “symmetric” in the sense that if $G(x_0,y,z)\lt r$ then both $y$ and $z$ lie in $B_G(x_0,r)$. Proposition 4 of Mustafa–Sims shows that these G-balls form a basis for a topology $\tau(G)$ on $X$, and moreover $\tau(G)$ coincides with the metric topology arising from the associated metric $d_G(x,y) = G(x,y,y) + G(x,x,y)$ [[2]](#references). In particular, every G-metric space is topologically equivalent to an ordinary metric space.

From this one deduces that convergence and continuity in G-metric spaces behave just as in metric spaces. A sequence $(x_n)\subset X$ is G-convergent to $x\in X$ if $x_n\to x$ in the $\tau(G)$-topology. Equivalently, as shown by Mustafa–Sims, $(x_n)$ converges in the G-sense to $x$ if and only if
$$
d_G(x_n, x) \xrightarrow{n\rightarrow\infty} 0,
$$
that is, it converges in the usual metric sense. In fact, the following are equivalent [[2]](#references):

- $(x_n)$ is G-convergent to $x$.
- $d_G(x_n,x)\to0$ as $n\to\infty$.
- $G(x_n,x_n,x)\to0$ as $n\to\infty$.
- $G(x_n,x_m,x)\to0$ as $n,m\to\infty$.

Thus a map $f:(X,G)\to (X',G')$ is G-continuous (continuous in $\tau(G)$) exactly when it is continuous as a map between the corresponding metric spaces. In particular, $G(x,y,z)$ itself is jointly continuous in all three variables.

One likewise defines G-Cauchy sequences: $(x_n)$ is G-Cauchy if for every $\varepsilon>0$ there exists $N$ such that 
$$
G(x_n, x_m, x_\ell) <\varepsilon \qquad \forall n,m,\ell \geq N.
$$
Mustafa–Sims show this is equivalent to $(x_n)$ being Cauchy in $(X,d_G)$. Hence $(X,G)$ is G-complete (every G-Cauchy sequence converges in $G$) if and only if the metric space $(X,d_G)$ is complete. Equivalently, G-completeness coincides with ordinary completeness. As a result, the usual consequences hold: closed subsets of a G-complete space are G-complete, and one obtains a Baire Category theorem in G-metric spaces (every complete G-metric space is non-meager).

Compactness is similarly characterized: a G-metric space is compact if it is G-complete and totally bounded (equivalently, $(X,d_G)$ is compact). In fact, the following are equivalent:

- $(X,G)$ is compact G-metric.
- $(X,\tau(G))$ is compact.
- $(X,d_G)$ is compact.
- Every sequence has a G-convergent subsequence.

G-metric spaces have been extensively used in fixed-point theory, often yielding generalizations or new forms of contraction-type theorems.

> **Theorem (G-Banach fixed point) [[4]](#references).** Suppose $(X,G)$ is G-complete and $T:X\to X$ satisfies a contractive condition in the G-metric, e.g.
> $$G(Tx,Ty,Tz)\leq k\,G(x,y,z), \qquad 0\le k<1,$$
> for all $x,y,z\in X$. Then $T$ has a unique fixed point in $X$.

### 2.1 Example: a G-metric not induced by a pairwise metric

Let $X$ be the vertex set of a connected, edge-weighted, undirected graph. For three vertices $x,y,z\in X$, define
$$
G(x,y,z)=\{\text{length of a minimum Steiner tree that connects } \{x,y,z\}\}.
$$
Equivalently: the minimum total weight of a connected subgraph whose vertex set contains $\{x,y,z\}$.

This $G$ satisfies the axioms:
- $G(x,x,x)=0$.
- If $x\neq y$, then $G(x,x,y)$ is the shortest-path distance between $x$ and $y$, hence $>0$.
- Symmetry is immediate (the definition depends only on the set $\{x,y,z\}$).
- A “rectangle-type” inequality holds: for any $a\in X$, the union of Steiner trees for $\{x,y,a\},\{x,a,z\},\{a,y,z\}$ is a connected subgraph spanning $\{x,y,z\}$ whose total length is at most the sum of the three lengths. Since $G(x,y,z)$ is the minimum such length,
  $$
  G(x,y,z)\le G(x,y,a)+G(x,a,z)+G(a,y,z).
  $$

If a G-metric were composed of ordinary metrics, it would be determined by the three pairwise distances. Steiner length is **not** determined solely by $d(x,y),d(y,z),d(z,x)$—graph topology matters.

## 3. Applications to operator-valued kernels 

Let $K_z(x,y):=K(x,y,z)$ be positive definite for each fixed $z\in X$. Fix a Hilbert space of functions over $z$ (e.g. $\mathcal{H}=L^2(\mu)$). 
Define an operator-valued kernel
$$
\mathbb{K}(x,y): \mathcal{H}\to\mathcal{H}, 
\qquad 
\bigl[\mathbb{K}(x,y)f\bigr](z)=K(x,y,z)\,f(z).
$$

$\mathbb{K}$ is a valid positive–definite (PD) operator-valued kernel iff 
$K_z$ is PD for $\mu$-a.e. $z$. Then all the vector-valued RKHS machinery
applies (representer theorem; kernel ridge/SVM for multi-output functions),
with predictors of the form
$$
f(\cdot)\in\mathcal{H}, 
\qquad 
f(\cdot)=\sum_{i}\alpha_i K(x_i,x',\cdot).
$$
Then, this expression 
$$
\sum_{i,j} c_i c_j K(x_i,x_j,\cdot)
$$
is exactly an element of this output space $\mathcal{H}$.

Before giving an example, recall the notion of a conditionally negative definite (CND) kernel.

> **Definition**: A symmetric kernel $K(x,y)$ is conditionally negative definite (CND) if for every finite set of points $x_1, \dots, x_n$ and scalars $c_1, \dots, c_n$ satisfying 
> $$ \sum_{i} c_i = 0,$$
> we have 
> $$ \sum_{ij} c_i c_j K(x_i, x_j) \leq 0. $$

Consider a Hilbert-type metric $d(x,y)=\|\Phi(x)-\Phi(y)\|$ induced by a map $\Phi:X\to\mathcal H$. Then we can define the G-metric
$$
G(x,y,z)\ :=\ \tfrac12\big(\|\Phi(x)-\Phi(y)\|^2+\|\Phi(y)-\Phi(z)\|^2+\|\Phi(z)-\Phi(x)\|^2\big).
$$
This satisfies the axioms, and for each fixed $z$ one can write
$$
K_z(x,y)=G(x,y,z)
= \tfrac12\|\Phi(x)-\Phi(y)\|^2\ +\ u_z(x)+u_z(y),
\quad u_z(x):=\tfrac12\|\Phi(x)-\Phi(z)\|^2.
$$
Since $\|\Phi(x)-\Phi(y)\|^2$ is CND (Schoenberg), and additive terms of the form $u_z(x)+u_z(y)$ vanish in the CND test (because $\sum c_i=0$), it follows that $K_z$ is CND. Therefore, for any $t>0$, 
$$
[\overline K(x,y) f](z) := e^{-t\, G(x,y,z)}\, f(z)
$$
defines an operator-valued PD kernel.

## 4. Possible future applications 

### 4.1 Triplet-based representation learning

In deep metric learning, we often use *triplet loss*:
$$
L = \max(0, d(f(x_a), f(x_p)) - d(f(x_a), f(x_n)) + \alpha)
$$
which explicitly considers triples (anchor, positive, negative).

A G-metric could encode this triplet geometry directly, rather than defining it indirectly through pairwise distances. One could design a model $G_\theta(x,y,z)$ that is symmetric and satisfies the G-metric axioms while learning consistent triplet relationships.

### 4.2 Higher-order relational learning

In graph representation learning or knowledge graphs, relationships are often ternary or higher, e.g., subject–predicate–object.
A G-metric provides a natural way to model triadic similarity or hypergraph distances.

### 4.3 Higher-order attention
In higher-order attention, the focus is on interactions among triples or tuples of elements:
$$
\operatorname{Attn}(x_i, x_j, x_k) = f(G(x_i, x_j, x_k))
$$
Here, the G-metric measures triadic coherence among the three embeddings, not just pairwise similarity. This contrasts with common approaches to higher-order attention that rely on tensor products [[5]](#references).
This captures contextual or relational dependencies (e.g., how three tokens jointly influence meaning).
It’s useful for relational reasoning, scene understanding, or multi-agent interactions, where relationships are inherently non-pairwise.

## Related work

- **Triplet Consistency and Metric Geometry in ML**: "Deep Metric Learning Beyond Pairwise Comparisons" (CVPR 2019)
- **Gromov–Wasserstein Learning**: Peyré et al., Foundations of Computational Optimal Transport, 2019
- **Hyperbolic / Non-Euclidean Embedding Learning**: Nickel & Kiela, Poincaré Embeddings for Hierarchical Representations (NeurIPS 2017)
- **Higher-Order Geometric Learning**: “Neural Hypergraph Learning” (AAAI 2021)

## References

[1] Agarwal, R.P., Karapınar, E., O’Regan, D., Roldán-López-de-Hierro, A.F. (2015). G-Metric Spaces. In: Fixed Point Theory in Metric Type Spaces. Springer, Cham. https://doi.org/10.1007/978-3-319-24082-4_3

[2] Mustafa, Zead, and Brailey Sims. "A new approach to generalized metric spaces." Journal of Nonlinear and Convex Analysis 7, no. 2 (2006): 289.

[3] Jleli, Mohamed, and Bessem Samet. "Remarks on G-metric spaces and fixed point theorems." Fixed Point Theory and Applications 2012, no. 1 (2012): 210.

[4] Alghamdi, Maryam A., and Erdal Karapınar. "G-β-ψ-contractive type mappings in G-metric spaces." Fixed Point Theory and Applications 2013, no. 1 (2013): 123.

[5] Omranpour, Soroush, Guillaume Rabusseau, and Reihaneh Rabbany. "Higher Order Transformers: Efficient Attention Mechanism for Tensor Structured Data." arXiv preprint arXiv:2412.02919 (2024).

