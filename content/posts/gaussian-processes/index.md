---
title: "Gaussian Processes"
date: 2025-09-24
tags: ["machine learning", "gaussian processes"]
author: "Daniel López Montero"
showToc: true
draft: true
description: "Analysis, Predictions and Experiments"
ShowWordCount: false
ShowReadingTime: true
comments: true
UseHugoToc: true
editPost:
    URL: "https://github.com/dani2442/dani2442.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---








Nice — let’s do this cleanly, step by step. I’ll derive **Gaussian process regression** as the **best linear estimator** (in mean-square-error sense). I’ll start with the zero-mean case (simpler), then show the nonzero-mean generalization and the connection to conditioning of a joint Gaussian.

---

# 1. Problem setup

* Training inputs: (X={x_i}_{i=1}^n).
* Latent function values at training inputs: (f = [f(x_1),\dots,f(x_n)]^\top).
* Test input: (x_*) with latent value (f_* = f(x_*)).
* Observations: (y_i = f(x_i) + \varepsilon_i), with noise (\varepsilon_i\stackrel{\text{iid}}{\sim}\mathcal N(0,\sigma^2)). So (y = f + \varepsilon).
* Prior on (f(\cdot)): a Gaussian process with mean (m(\cdot)) and covariance kernel (k(\cdot,\cdot)). For now assume (m(\cdot)\equiv 0); we’ll generalize later.

Define covariance matrices/vectors:

* (K \in\mathbb R^{n\times n}) with (K_{ij}=k(x_i,x_j)).
* (k_* \in\mathbb R^n) with ((k_*)*i = k(x_i,x**) = \operatorname{Cov}(y_i,f_*)) (note noise on (y) is independent of (f_*)).
* (k_{**} = k(x_*,x_*) = \operatorname{Var}(f_*)).
* Covariance of the observed vector (y): (\mathrm{Cov}(y,y)=K+\sigma^2 I \equiv C_{yy}.)

Goal: find the linear estimator (\hat f_* = w^\top y) (allowing a constant offset later) that minimizes mean squared error (MSE)
[
\mathrm{MSE}(w)=\mathbb E\big[(f_* - w^\top y)^2\big].
]

---

# 2. Minimizing MSE (zero-mean prior)

Write the MSE and expand (expectations are over the joint prior of (f) and noise):

[
\begin{aligned}
\mathrm{MSE}(w)
&= \mathbb E[f_*^2] - 2,\mathbb E[f_*, w^\top y] + \mathbb E[w^\top y y^\top w] \
&= k_{**} - 2, w^\top \mathbb E[y f_*] + w^\top \mathbb E[y y^\top] w.
\end{aligned}
]

But (\mathbb E[y f_*] = k_*) and (\mathbb E[yy^\top] = C_{yy} = K + \sigma^2 I). So

[
\mathrm{MSE}(w) = k_{**} - 2, w^\top k_* + w^\top C_{yy} w.
]

This is a quadratic function of (w). Differentiate w.r.t. (w) and set gradient to zero:

[
\frac{\partial}{\partial w}\mathrm{MSE}(w) = -2 k_* + 2 C_{yy} w = 0
\quad\Rightarrow\quad C_{yy} w^* = k_*.
]

Assuming (C_{yy}) is invertible (typical if (\sigma^2>0) or (K) is full rank),

[
\boxed{w^* = C_{yy}^{-1} k_* = (K+\sigma^2 I)^{-1} k_*.}
]

Thus the best linear estimator is

[
\boxed{\hat f_* = {w^*}^\top y = k_*^\top (K+\sigma^2 I)^{-1} y.}
]

This is exactly the usual GP posterior mean (for zero prior mean).

The minimal MSE (plug (w^*) back in) is

[
\begin{aligned}
\mathrm{MSE}*{\min}
&= k*{**} - {k_*}^\top C_{yy}^{-1} k_*.
\end{aligned}
]

This equals the GP posterior variance at (x_*). So the minimal achievable MSE by any linear estimator equals the posterior variance.

---

# 3. Orthogonality (projection) interpretation

An equivalent, illuminating derivation uses the orthogonality principle (linear projection): the error (e = f_* - w^\top y) of the best linear estimator must be uncorrelated with the data used in the estimator:

[
\mathbb E[e, y] = 0 \quad\Rightarrow\quad \mathbb E[(f_* - w^\top y) y] = 0.
]

Thus (\mathbb E[f_* y] - \mathbb E[y y^\top] w = 0), i.e. (k_* - C_{yy} w = 0), giving the same solution (w = C_{yy}^{-1} k_*). So the GP predictor is the linear projection of (f_*) onto the subspace spanned by the observed (y).

---

# 4. Joint Gaussian conditioning (why GP posterior mean is linear)

If you prefer the standard GP route: the joint distribution of (y) and (f_*) is Gaussian:

[
\begin{bmatrix} y \ f_* \end{bmatrix}
\sim
\mathcal N!\left(0,,
\begin{bmatrix}
C_{yy} & k_* \
k_*^\top & k_{**}
\end{bmatrix}
\right).
]

Conditional expectation for a multivariate Gaussian yields

[
\mathbb E[f_* \mid y] = k_*^\top C_{yy}^{-1} y,
\qquad
\operatorname{Var}(f_* \mid y) = k_{**} - k_*^\top C_{yy}^{-1} k_*.
]

So the posterior mean equals the linear estimator we derived by minimizing MSE. For Gaussian processes, the posterior mean is both the Bayes (minimum mean squared error over all measurable estimators) and — as we've shown — the best linear estimator.

Important conceptual note: for a Gaussian prior the **optimal** estimator under squared loss is the conditional mean, and because of Gaussianity that conditional mean is linear in (y). For non-Gaussian priors the best linear estimator (projection) is still (k_*^\top C_{yy}^{-1} y) but it will not, in general, equal the conditional mean.

---

# 5. Nonzero mean function

If the GP has mean function (m(\cdot)), write (m_X = [m(x_1),\dots,m(x_n)]^\top) and (m_* = m(x_*)). The model is (y = m_X + (f - m_X) + \varepsilon). Working with centered quantities (y - m_X) and (f_* - m_*) (which have zero mean) gives the estimator

[
\boxed{ \mathbb E[f_* \mid y] = m_* + k_*^\top (K+\sigma^2 I)^{-1} (y - m_X). }
]

So you subtract the prior mean from observations, apply the same weight matrix, and add the prior mean back at the test point.

---

# 6. Multiple test points (vector form)

For test locations (X_*) with latent vector (f_*), define (K_{*n}=K(X_*,X)), (K_{**}=K(X_*,X_*)). Then

[
\boxed{\mathbb E[f_* \mid y] = m_{*} + K_{*n} (K+\sigma^2 I)^{-1}(y-m_X),}
]
[
\operatorname{Cov}(f_* \mid y) = K_{**} - K_{*n}(K+\sigma^2 I)^{-1}K_{n*}.
]

---

# 7. Summary (the punchlines)

* Minimizing (\mathbb E[(f_* - w^\top y)^2]) over linear maps (w^\top y) gives (w^*=(K+\sigma^2 I)^{-1}k_*) and estimator (k_*^\top(K+\sigma^2 I)^{-1}y).
* This estimator equals the GP posterior mean (conditioning a joint Gaussian).
* The minimal MSE equals the posterior variance (k_{**}-k_*^\top(K+\sigma^2 I)^{-1}k_*).
* With nonzero mean (m(\cdot)), replace (y) by (y-m_X) and add (m_*) back.

---

If you want, I can:

* show a compact matrix algebra derivation for multiple test points,
* demonstrate with a tiny numerical example (compute weights and predictions), or
* connect this to kriging / BLUE and the unbiasedness constraint.

Which would you like next?
