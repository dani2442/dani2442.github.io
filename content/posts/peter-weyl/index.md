---
title: "Harmonic Analysis: Peter-Weyl Theorem and Machine Learning"
date: 2025-10-17
tags: ["machine learning", "mixture of gaussians", "harmonic analysis"]
categories: ["machine learning", "mathematics", "harmonic analysis"]
author: "Daniel López Montero"
showToc: true
draft: true
description: ""
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


## 1. Preliminaries

We start by introducing some basic notions from representation theory and harmonic analysis on compact groups. 

Let us denote by $G$ an arbitrary locally compact group with a fixed Haar measure $\mu$. 

> **Definition (Representation).** A (unitary) representation of $G$ is a continuous homomorphism  from $G$ to the unitary group $\mathfrak{U}(\mathcal{H}_\pi)$ on a complex Hilbert space $\mathcal{H}_\pi$.

$$ \pi(gh) = \pi(g)\pi(h), \quad \forall g,h\in G. $$

*We denote $d_\pi := \dim(\mathcal{H}_\pi)$ the dimension of the representation.*

The condition that $\pi$ is continuous means that $g\mapsto \pi(g)u$ is continuous for all $u\in \mathcal{H}_\pi$.

Sometimes, people use the more general $\mathfrak{U}(\mathcal{H}_\pi)\subset\mathrm{GL}(\mathcal{H}_\pi)$ instead. However, they are equivalent for compact groups.

> **Example.** If we choose $\mathcal{H}_\pi:=L^2(G)$, then the left regular representation $\pi: G\rightarrow \mathcal{B}(\mathcal{H}_\pi)$ is defined by $\pi(g)f(x) = f(g^{-1}x)$ for all $f\in L^2(G)$.

> **Definition (Equivalence of representations).** Two representations $\pi_1: G\rightarrow \mathfrak{U}(\mathcal{H}_1)$ and $\pi_2: G\rightarrow \mathfrak{U}(\mathcal{H}_2)$ are said to be equivalent if there exists a unitary operator $U:\mathcal{H}_1\rightarrow \mathcal{H}_2$ such that $\pi_1(g) = U\pi_2(g)U^*$ for all $g\in G$. We denote this by $\pi_1\cong \pi_2$, and let $[\pi]$ be the equivalence class of $\pi$. 

> **Definition (Irreducible representation).** A representation $\pi: G\rightarrow \mathfrak{U}(\mathcal{H}_\pi)$ is said to be irreducible if the only closed subspaces of $\mathcal{H}_\pi$ that are invariant under $\pi(g)$ for all $g\in G$ are $\{0\}$ and $\mathcal{H}_\pi$ itself.

We denote by $\widehat{G}$ the set of equivalence classes of irreducible unitary representations of $G$.

If $G$ is abelian, then every irreducible representation is one-dimensional. In particular, $\widehat{G}$ coincides with the character group (Pontryagin dual) of $G$.

Let $\pi, \eta$ be two representations of $G$, then $\pi\otimes \eta$ is defined on the tensor product space $\mathcal{H}_\pi \otimes \mathcal{H}_\eta$ by 
$$ (\pi\otimes \eta)(g)(\xi, \zeta) = \pi(g)\xi \otimes \eta(g)\zeta, \quad \forall g\in G. $$

If $n\in \mathbb{Z}^+ \cup \{\infty\}$, then $n\cdot \pi = \oplus_{i=1}^n \pi$ is the direct sum of $n$ copies of $\pi$.


> **Definition.** If $\pi$ is a representation of $G$, then
$$
\pi(G)' := \{A\in \mathcal{B}(\mathcal{H}_\pi) : A\pi(g) = \pi(g)A, \forall g\in G\}
$$
is a self-adjoint subalgebra of $\mathcal{B}(\mathcal{H}_\pi)$ called the commutant of $\pi(G)$.

> **Theorem 1.** Suppose that $\pi$ is a representation of $G$. Then the following are equivalent:
> 1. $\pi$ is irreducible.
> 2. $\pi(G)' = \mathbb{C}I$.
> 3. Every non-zero $\xi\in \mathcal{H}_\pi$ is cyclic, i.e. $\overline{\mathrm{span}}\{\pi(g)\xi : g\in G\} = \mathcal{H}_\pi$.

## 2. Peter-Weyl Theorem

In this section we specialize to **compact groups**. Recall that compact groups are characterized by the fact that Haar measure satisfies $\mu(G)<\infty$. It is customary to normalize Haar measure on a compact group by choosing the unique measure such that
$$
\mu(G)=1.
$$
With this normalization, for $f\in L^1(G)$ we write
$$
\int_G f(g)\,dg.
$$

Assume that $\pi$ is a finite-dimensional unitary representation of $G$ on $\mathcal{H}_\pi$. Choose an orthonormal basis
$$
\mathcal{B}=\{e_1,\dots,e_{d_\pi}\}
\quad\text{of }\mathcal{H}_\pi.
$$
For each $g\in G$, the matrix of $\pi(g)$ with respect to $\mathcal{B}$ has $(i,j)$-entry. The functions
$$
\varphi_{ij}^{\pi}(g):=\langle e_i,\pi(g)e_j\rangle,\qquad 1\le i,j\le d_\pi,
$$
are called the **coordinate functions** of $\pi$.

Notice that $\varphi_{ij}^{\pi}\in C(G)$, and more generally for any $\xi,\eta\in \mathcal{H}_\pi$ the function
$$
g\longmapsto \langle \xi,\pi(g)\eta\rangle
$$
is continuous.

> **Definition (The coefficient space $E$).** Let $E_G$ be the linear span of all functions of the form
$$
g\mapsto \langle \xi,\pi(g)\eta\rangle,
$$
where $\pi$ ranges over all **irreducible** unitary representations of $G$, and $\xi,\eta$ range over $\mathcal{H}_\pi$.

Since every finite-dimensional representation is a direct sum of irreducibles, it follows in particular that every matrix coefficient coming from an arbitrary finite-dimensional representation belongs to $E$.

> **Remark (Abelian case and trigonometric polynomials).**  
When $G$ is abelian, the functions in $E$ are often called **trigonometric polynomials**. For example, if
$$
G=\mathbb{T}=\mathbb{R}/2\pi\mathbb{Z},
$$
then every $f\in E$ has the form
$$
f(\theta)=\sum_{n=-N}^N c_n e^{in\theta}
\;=\;
\sum_{n=0}^N \bigl(a_n\cos(n\theta)+b_n\sin(n\theta)\bigr).
$$

> **Remark (Self-adjointness and inversion).**  
It is clear that $E$ is **self-adjoint** in the sense that if $f\in E$, then $\overline{f}\in E$. In terms of coordinate functions, one checks
> $$ \overline{\varphi_{ij}^{\pi}(g)}=\varphi_{ji}^{\pi}(g).$$
It is also true that if $f\in E$, then the function $\widetilde f$ defined by
$$
\widetilde f(g):=f(g^{-1})
$$
belongs to $E$ as well. One way to see this is by introducing the **conjugate Hilbert space** $\overline{\mathcal{H}}_\pi$ and the conjugate representation $\overline\pi$; then
$$
\varphi_{ij}^{\overline\pi}(g)=\varphi_{ij}^{\pi}(g^{-1}).
$$
This justifies the terminology “coordinate functions”.

### 2.2. Hilbert-Schmidt norms

> **Definition (Hilbert-Schmidt norm).** Let $M_n$ be the complex $n\times n$ matrices. If $A=(a_{ij})\in M_n$, define the Hilbert-Schmidt norm by
> $$
\|A\|_{\mathrm{HS}}^2=\sum_{i,j}|a_{ij}|^2.
$$

> **Remark.** If $A\in M_n$, then
> $$
\|A\|_{\mathrm{HS}}^2=\mathrm{tr}(AA^*).
$$
In particular, if $B=UAU^*$ for a unitary matrix $U$, then $\|A\|_{\mathrm{HS}}=\|B\|_{\mathrm{HS}}$.
Hence, if $\pi$ is finite-dimensional, the value $\|\pi(g)\|_{\mathrm{HS}}$ depends only on the equivalence class $[\pi]$.

### 2.3. Statement of the Peter-Weyl theorem

Our goal is to prove the following result.

> **Theorem (Peter-Weyl).** Let $G$ be a compact group.
> 1. Every irreducible unitary representation of $G$ is **finite-dimensional**.
> 2. If $\lambda$ is the left regular representation of $G$ on > $L^2(G)$, then
>   $$
   \lambda \;\cong\; \bigoplus_{[\pi]\in \widehat{G}} d_\pi\cdot \pi.  $$
> 3. (**Separation of points**) Given $g\in G$ with $g\ne e$, there exists $[\pi]\in\widehat G$ such that
>   $$  \pi(g)\ne I.  $$
> 4. The space $E$ is **dense** in $C(G)$ (hence also dense in $L^p(G)$ for all $1\le p<\infty$).
> 5. (**Plancherel / Parseval identity**) If $f\in L^2(G)$, then
> $$
   \|f\|_2^2 =
   \sum_{[\pi]\in\widehat G} d_\pi\,
   \mathrm{tr}\bigl(\widehat f(\pi)\,\widehat f(\pi)^*\bigr) =
   \sum_{[\pi]\in\widehat G} d_\pi \|\widehat f(\pi)\|_{\mathrm{HS}}^2,
   $$
   where the operator-valued Fourier transform $\widehat f(\pi)$ is defined by
   $$
   \widehat f(\pi)\;:=\;\int_G f(g)\,\pi(g)\,dg
   \quad\in\mathcal{B}(\mathcal{H}_\pi).
   $$

We will prove this theorem using a sequence of preliminary results.

---

### 2.4. Intertwiners and invariant subspaces

> **Lemma 1.** Let $\pi$ and $\eta$ be representations of a locally compact group $G$. If $A:\mathcal{H}_\pi\to\mathcal{H}_\eta$ is a bounded operator satisfying
> $$
A\pi(g)=\eta(g)A,\qquad \forall g\in G, $$
> then
> $$
A^*A\,\eta(g)=\eta(g)\,A^*A,\qquad \forall g\in G. $$

**Proof.** For $\xi,\zeta\in\mathcal{H}_\eta$ and $g\in G$,
$$
\langle A^*A\eta(g)\xi,\zeta\rangle
=\langle A\eta(g)\xi,A\zeta\rangle
=\langle \eta(g)A\xi,A\zeta\rangle
=\langle A\xi,\eta(g^{-1})A\zeta\rangle
=\langle \eta(g)A^*A\xi,\zeta\rangle,
$$
which implies the claim. $\square$

> **Lemma 2.** Suppose $\pi$ and $\eta$ are representations of a locally compact group $G$, with $\pi$ irreducible. If $A:\mathcal{H}_\pi\to\mathcal{H}_\eta$ is a nonzero bounded operator satisfying
> $$
A\pi(g)=\eta(g)A,\qquad \forall g\in G, $$
> then:
> 1. $A\mathcal{H}_\pi$ is a **closed** $\eta$-invariant subspace of $\mathcal{H}_\eta$;
> 2. $\pi\cong \eta|_{A\mathcal{H}_\pi}$.

**Proof.** By the Lemma 1, $A^*A$ belongs to the commutant $\eta(G)'$. By Theorem 1, the $A^*A = \lambda I$. Thus $B:=\lambda^{-1/2}A$ is an isometry, so it is unitary from $\mathcal{H}_\pi$ onto $A\mathcal{H}_\pi$, which is closed as well.

Again by Lemma 1, $BB^*\in \eta(G)'$. Since $BB^*$ is the orthogonal projection onto $A\mathcal{H}_\pi = B\mathcal{H}_\pi$, it follows that $A\mathcal{H}_\pi$ is $\eta$-invariant. $\square$

---

### 2.5. Orthogonality relations

The next proposition is the key orthogonality statement for matrix coefficients, and it depends crucially on compactness of $G$.

> **Proposition (Schur orthogonality).** Let $\pi$ and $\eta$ be irreducible representations of a compact group $G$. Fix orthonormal bases $\{e_i^\pi\}_{i=1}^{d_\pi}$ of $\mathcal{H}_\pi$ and $\{e_k^\eta\}_{k=1}^{d_\eta}$ of $\mathcal{H}_\eta$, and define
> $$ \varphi_{ij}^\pi(g)=\langle e_i^\pi,\pi(g)e_j^\pi\rangle, \qquad \varphi_{kl}^\eta(g)=\langle e_k^\eta,\eta(g)e_l^\eta\rangle. $$
> 1. If $\pi\not\cong \eta$, then
>   $$
   \int_G \varphi_{ij}^\pi(g)\,\varphi_{kl}^\eta(g)\,dg=0
   \quad\text{for all }i,j,k,l.    $$
> 2. If $\pi$ is finite-dimensional, then
>   $$  \int_G \varphi_{ij}^\pi(g)\,\varphi_{kl}^\pi(g)\,dg    =   \frac{\delta_{ik}\delta_{jl}}{d_\pi},   \quad\text{for all }i,j,k,l.   $$

**Proof.** Let $B:\mathcal{H}_\pi\to\mathcal{H}_\eta$ be bounded and define
$$
A=\int_G \eta(g)\,B\,\pi(g^{-1})\,dg.
$$
Then for $r\in G$,
$$
A\pi(r)
=
\int_G \eta(g)B\pi(g^{-1}r)\,dg
=
\int_G \eta(rg)B\pi(g^{-1})\,dg
=
\eta(r)A,
$$
so $A$ intertwines $\pi$ and $\eta$.

If $\pi\not\cong\eta$, the Lemma 2 must be $A=0$. Choosing $B$ to be rank-one operators of the form
$$
B_{ij}(\xi)=\langle \xi,e_j^\pi\rangle e_l^\eta,
$$
and testing against basis vectors yields the integral identity in (1) as follows:
$$
\begin{aligned}
0 &= \langle A e_j^\pi, e_l^\eta \rangle = 
\int_G \langle B_{ij} \pi(g^{-1}) e_j^\pi, \eta(g^{-1}) e_l^\eta \rangle dg \\ &= 
\int_G \langle \pi(g^{-1}) e_i^\pi, e_j^\pi \rangle \langle e_l^\eta, \eta(g^{-1}) e_k^\eta \rangle dg \\ &=
\int_G \varphi_{ij}^\pi(g^{-1}) \varphi_{kl}^\eta(g^{-1}) dg 
\end{aligned}
$$

Now assume $d_\pi<\infty$. By Lemma 2,
$$
A = \int_G \pi(g) B \pi(g^{-1}) dg = \lambda I \qquad \forall B\in \mathcal{B}(\mathcal{H}_\pi).
$$
Taking traces on both sides gives
$$
\begin{aligned}
\mathrm{tr}(A) &= \sum_{i=1}^{d_\pi} \langle A e_i^\pi, e_i^\pi \rangle =  \int_G \sum_{i=1}^{d_\pi} \langle B \pi(g^{-1}) e_i^\pi, \pi(g^{-1}) e_i^\pi \rangle dg \\ &=
\int_G \mathrm{tr}(B \pi(g^{-1}) \pi(g)) dg = \mathrm{tr}(B),
\end{aligned}
$$
Since $\mathrm{tr}(A) = \mathrm{tr}(\lambda I )= \lambda d_\pi$ and $\mathrm{tr}(B_{jl}) = \delta_{jl}$, we have $\lambda = \delta_{jl}/d_\pi$ when $B = B_{jl}$. On the other hand, 
$$
\int_G \varphi_{ij}^\pi(g)\,\overline{\varphi_{kl}^\pi(g)}\,dg = \langle A e_i^\pi, e_k^\pi \rangle = \lambda \langle e_i^\pi, e_k^\pi \rangle = \frac{\delta_{ik}\delta_{jl}}{d_\pi}
$$
$\square$

---

### 2.6. Proof of the Peter--Weyl theorem

Let $E_{\mathrm{fin}}\subseteq E$ denote the subspace spanned by matrix coefficients coming from **finite-dimensional** irreducible representations of $G$.

We first show that it suffices to prove that $E_{\mathrm{fin}}$ is dense in $C(G)$.

Assume $E_{\mathrm{fin}}$ is dense in $C(G)$. Since $C(G)$ is dense in $L^2(G)$, the orthogonality relations imply that $L^2(G)$ admits an orthonormal basis consisting of the normalized coefficients
$$
\sqrt{d_\pi}\,\varphi_{ij}^\pi,\qquad [\pi]\in\widehat G,\ 1\le i,j\le d_\pi.
$$

Indeed, if $\pi$ is irreducible and $d_\pi=\infty$, then Proposition (Schur orthogonality) forces all coefficients $\varphi_{ij}^\pi$ to be orthogonal to $E_{\mathrm{fin}}$; by density they must vanish, hence $\pi$ cannot occur. This proves item (1): all irreducibles are finite-dimensional.

Now, for each finite-dimensional irreducible $\pi$, Proposition (Schur orthogonality) shows that the $d_\pi^2$ functions $\{\sqrt{d_\pi}\,\varphi_{ij}^\pi\}_{i,j}$ form an orthonormal basis for a subspace $H_\pi\subseteq L^2(G)$.

Observe that
$$
\varphi_{ij}^\pi(g^{-1}t)
=
\langle e_i^\pi,\pi(g^{-1}t)e_j^\pi\rangle
=
\sum_{k=1}^{d_\pi} \varphi_{ik}^\pi(g^{-1})\,\varphi_{kj}^\pi(t).
$$
For each fixed $j$, define an operator $A_j:\mathcal{H}_\pi\to H_\pi$ by
$$
A_j(e_i^\pi)=\varphi_{ij}^\pi.
$$
Using the identity above, one checks that $A_j$ intertwines $\pi$ with the left regular representation $\lambda$ restricted to $H_\pi$:
$$
\lambda(g)\,A_j = A_j\,\pi(g).
$$
By the previous lemma, $\pi$ appears as a subrepresentation of $\lambda$ on $H_\pi$, and moreover
$$
\lambda|_{H_\pi}\cong d_\pi\cdot \pi.
$$
Summing over $[\pi]\in\widehat G$ yields item (2):
$$
\lambda \cong \bigoplus_{[\pi]\in\widehat G} d_\pi\cdot\pi.
$$

At this point we know that the functions $\{\sqrt{d_\pi}\,\varphi_{ij}^\pi\}$ form an orthonormal basis of $L^2(G)$, so every $f\in L^2(G)$ has an expansion
$$
f=\sum_{[\pi]\in\widehat G}\ \sum_{i,j=1}^{d_\pi} c_{ij}^\pi\,\sqrt{d_\pi}\,\varphi_{ij}^\pi,
$$
with
$$
\|f\|_2^2=\sum_{[\pi]\in\widehat G}\ \sum_{i,j=1}^{d_\pi} |c_{ij}^\pi|^2.
$$

Moreover,
$$
c_{ij}^\pi
=
\int_G f(g)\,\sqrt{d_\pi}\,\varphi_{ij}^\pi(g)\,dg
=
\sqrt{d_\pi}\int_G f(g)\,\langle e_i^\pi,\pi(g)e_j^\pi\rangle\,dg
=
\sqrt{d_\pi}\,\langle e_i^\pi,\widehat f(\pi)e_j^\pi\rangle.
$$
Therefore
$$
\|f\|_2^2
=
\sum_{[\pi]\in\widehat G} d_\pi\sum_{i,j=1}^{d_\pi} |\langle e_i^\pi,\widehat f(\pi)e_j^\pi\rangle|^2
=
\sum_{[\pi]\in\widehat G} d_\pi\,\|\widehat f(\pi)\|_{\mathrm{HS}}^2,
$$
which is item (5).

Item (3) follows from item (4): if $E$ were dense but did not separate points, it would contradict Stone--Weierstrass.

Thus it only remains to prove:

> **Claim.** $E_{\mathrm{fin}}$ is dense in $C(G)$.

We prove this using Hilbert--Schmidt operators and an approximate identity.

---

### 2.7. Hilbert--Schmidt operators and convolution by class functions

Let $(X,\nu)$ be a measure space and let $K\in L^2(X\times X,\nu\otimes\nu)$. Define an operator $T:L^2(X)\to L^2(X)$ by
$$
Tf(x)=\int_X K(x,y)f(y)\,d\nu(y).
$$
Such operators are called **Hilbert--Schmidt operators**; they are compact, and if $K(x,y)=\overline{K(y,x)}$, then $T$ is self-adjoint. In particular, each eigenspace
$$
H_\alpha=\{f\in L^2(X): Tf=\alpha f\}
$$
is finite-dimensional.

In our case we take $X=G$ with normalized Haar measure and $K$ continuous.

Now fix a function $k\in C(G)$ satisfying
$$
k(g)=k(g^{-1}),\qquad \forall g\in G.
$$
Define the convolution operator $T:L^2(G)\to L^2(G)$ by
$$
Tf = f*k,
\qquad
Tf(g)=\int_G f(r)\,k(r^{-1}g)\,dr.
$$
Equivalently, $T$ is a Hilbert--Schmidt operator with kernel
$$
K(g,r)=k(r^{-1}g),
$$
which is symmetric in the appropriate sense, so $T$ is self-adjoint and compact.

> **Lemma.** Let $k$ and $T$ be as above. For each $\alpha\in\mathbb{R}\setminus\{0\}$, the eigenspace
> $$ H_\alpha=\{f\in L^2(G): Tf=\alpha f\} $$
> is contained in $E_{\mathrm{fin}}$.

**Proof.** Since $T$ is compact and self-adjoint, $H_\alpha$ is finite-dimensional and consists of continuous functions (because $k$ is continuous and $Tf$ is continuous whenever $f\in L^2(G)$).

Moreover, $H_\alpha$ is invariant under the left regular representation. Indeed,
$$
T(\lambda(g)f)
=
(\lambda(g)f)*k
=
\lambda(g)(f*k)
=
\lambda(g)(Tf),
$$
so if $Tf=\alpha f$, then $T(\lambda(g)f)=\alpha\,\lambda(g)f$.

Let $\{f_1,\dots,f_r\}$ be an orthonormal basis of $H_\alpha$. Define continuous functions
$$
\theta_{ij}(g):=\langle \lambda(g)f_i,f_j\rangle.
$$
Since $\lambda(g)f_i\in H_\alpha$, we may expand
$$
(\lambda(g)f_i)(t)=\sum_{k=1}^r \theta_{ki}(g)\,f_k(t),
$$
and the map $g\mapsto (\theta_{ij}(g))_{i,j}$ defines a finite-dimensional unitary representation $\rho$ of $G$ on $\mathbb{C}^r$. In particular, each $f_i$ lies in the span of matrix coefficients of a finite-dimensional representation, hence in $E_{\mathrm{fin}}$. $\square$

> **Lemma.** Let $k$ and $T$ be as above. Then for every $f\in L^2(G)$ we have
> $$ Tf\in \overline{E_{\mathrm{fin}}}^{\|\cdot\|_2}. $$
> In particular, $Tf$ can be approximated in $L^2$ by elements of $E_{\mathrm{fin}}$.

**Proof.** Let $\{ \phi_n \}$ be a complete orthonormal set of eigenvectors for the nonzero eigenspaces of $T$, with eigenvalues $\alpha_n\ne 0$:
$$
T\phi_n=\alpha_n\phi_n.
$$
Then by the spectral theorem, any $f\in L^2(G)$ can be written as
$$
f=\sum_{n} c_n\phi_n + f_0,
\qquad Tf_0=0,
$$
with $\sum_n |c_n|^2<\infty$. Hence
$$
Tf=\sum_n \alpha_n c_n \phi_n.
$$
Approximating this series by partial sums yields $L^2$-convergence, and each $\phi_n\in E_{\mathrm{fin}}$ by the previous lemma. Thus $Tf$ is in the $L^2$-closure of $E_{\mathrm{fin}}$. $\square$

---

### 2.8. Approximate identities and density of $E_{\mathrm{fin}}$

We now use the existence of an approximate identity in $C(G)$.

> **Proposition (Approximate identity).** If $G$ is compact, there exists a net $\{k_U\}\subset C(G)$ indexed by neighborhoods $U$ of $e\in G$ such that:
> 1. $k_U\ge 0$, $\mathrm{supp}(k_U)\subseteq U$,
> 2. $\displaystyle \int_G k_U(g)\,dg = 1$,
> 3. $k_U(g)=k_U(g^{-1})$,
> 4. for every $f\in C(G)$, both $k_U*f\to f$ and $f*k_U\to f$ uniformly on $G$.

**Proof.** For each neighborhood $U$ of $e$, choose $k_U\in C(G)$ nonnegative with $\mathrm{supp}(k_U)\subseteq U$, $k_U(g)=k_U(g^{-1})$, and normalize so $\int_G k_U=1$.

Fix $f\in C(G)$ and $\varepsilon>0$. By uniform continuity of $f$ (compactness of $G$), there exists a neighborhood $W$ of $e$ such that
$$
|f(tg)-f(g)|<\varepsilon,\qquad \forall g\in G,\ \forall t\in W.
$$
If $U\subseteq W$, then for all $g\in G$,
$$
|(k_U*f)(g)-f(g)|
=
\left|\int_G k_U(t)\bigl(f(tg)-f(g)\bigr)\,dt\right|
\le
\int_G k_U(t)\,\varepsilon\,dt
=\varepsilon.
$$
Thus $k_U*f\to f$ uniformly. A similar argument yields $f*k_U\to f$ uniformly. $\square$

Now let $T_U$ be the convolution operator
$$
T_Uf := k_U*f.
$$
Each $k_U$ satisfies the hypotheses of the Hilbert--Schmidt lemmas above, and $T_Uf\to f$ uniformly for every $f\in C(G)$.

By the previous lemma, $T_Uf\in \overline{E_{\mathrm{fin}}}^{\|\cdot\|_2}$ for every $f\in L^2(G)$. In particular, if $f\in C(G)$ then $T_Uf$ is continuous and can be approximated by elements of $E_{\mathrm{fin}}$; since $T_Uf\to f$ uniformly, we conclude:

$$
\overline{E_{\mathrm{fin}}}^{\|\cdot\|_\infty}=C(G).
$$

This proves the claim, and therefore completes the proof of the Peter--Weyl theorem. $\blacksquare$

## References

[0] The PeterWeyl Theorem for Compact Groups by Dana P. Williams https://math.dartmouth.edu/~dana/bookspapers/pw.pdf?utm_source=chatgpt.com


[1] Elias M. Stein, *Topics in Harmonic Analysis Related to the Littlewood--Paley Theory*, Annals of Mathematics Studies, Princeton University Press.

[2] Walter Rudin, *Fourier Analysis on Groups*, Wiley Classics Library, John Wiley & Sons.

[3] Gerald B. Folland, *A Course in Abstract Harmonic Analysis*, Studies in Advanced Mathematics, CRC Press.