---
title: "Gaussian Processes"
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


Gaussian Processes regression is one of the key algorithms within machine learning. With this blog post I want to appreciate the beautiful mathematical theory behind that back them up. I will divide this post in theory and practice, including a sample code.

One of the key advantages of Gaussian Processes vs Deep Learning methods is that it inherently provide with confidence intervals and a higher level of interpretability. However, it comes with a hidden cost, it is has a very wide variety of hyperparameters that are far from easy to configure, i.e., only the kernel selection is very challenging. Understanding and having a good intuition in the inner workings of this algorithm is key to make the most of it. 

# 1. Theory

Problem setup

- Training inputs: $X=\{x_i\}_{i=1}^n$.
- Latent function values at training inputs: $f = [f(x_1),\dots,f(x_n)]^\top$.
- Test input: $x_*$ with latent value $f_* = f(x_*)$.
- Observations: $y_i = f(x_i) + \varepsilon_i$, with noise $\varepsilon_i\stackrel{\text{iid}}{\sim}\mathcal N(0,\sigma^2)$. So $y = f + \varepsilon$.
- Prior on $f(\cdot)\sim \mathcal{GP}$: a Gaussian process with mean $m(\cdot)$ and covariance kernel $k(\cdot,\cdot)$. For now assume $m(\cdot)\equiv 0$; we’ll generalize later.

Define covariance matrices/vectors:

- $K \in\mathbb R^{n\times n}$ with $K_{ij}=k(x_i,x_j)$.
- $k_* \in\mathbb R^n$ with $(k_*)_i = k(x_i,x_*) = \operatorname{Cov}(y_i,f_*)$. Note noise on $y$ is independent of $f_*$.
- $k_{**} = k(x_*,x_*) = \operatorname{Var}(f_*)$.
- Covariance of the observed vector $y$: $\mathrm{Cov}(y,y)=K+\sigma^2 I \equiv C_{yy}$.

## (Option 1) Best Linear Unbaised Estimator (BLUE)

Our goal is to find the linear estimator $\hat f_* := w^\top y$ that minimizes mean squared error (MSE)
$$
\mathrm{MSE}(w)=\mathbb E\big[(f_* - w^\top y)^2\big].
$$
Write the MSE and expand (expectations are over the joint prior of $f$ and noise):

$$
\begin{aligned}
\mathrm{MSE}(w)
&= \mathbb E[f_*^2] - 2,\mathbb E[f_*, w^\top y] + \mathbb E[w^\top y y^\top w] \\
&= k_{**} - 2, w^\top \mathbb E[y f_*] + w^\top \mathbb E[y y^\top] w.
\end{aligned}
$$

But $\mathbb E[y f_*] = k_*$ and $\mathbb E[yy^\top] = C_{yy} = K + \sigma^2 I$. So

$$
\mathrm{MSE}(w) = k_{**} - 2, w^\top k_* + w^\top C_{yy} w.
$$

This is a quadratic function of $w$. Differentiate w.r.t. $w$ and set gradient to zero:

$$
\frac{\partial}{\partial w}\mathrm{MSE}(w) = -2 k_* + 2 C_{yy} w = 0
\quad\Rightarrow\quad C_{yy} w^* = k_*.
$$

Assuming $C_{yy}$ is invertible (typical if $\sigma^2>0$ or $K$ is full rank),

$$
\boxed{w^* = C_{yy}^{-1} k_* = (K+\sigma^2 I)^{-1} k_*.}
$$

Thus the best linear estimator is

$$
\boxed{\hat f_* = {w^*}^\top y = k_*^\top (K+\sigma^2 I)^{-1} y.}
$$

This is exactly the usual GP posterior mean (for zero prior mean).

The minimal MSE (plug $w^*$ back in) is

$$
\begin{aligned}
\mathrm{MSE}_{\min}
&= k_{**} - {k_*}^\top C_{yy}^{-1} k_*.
\end{aligned}
$$

This equals the GP posterior variance at $x_*$. So the minimal achievable MSE by any linear estimator equals the posterior variance.


## (Option 2) Orthogonality interpretation

An equivalent, illuminating derivation uses the orthogonality principle (linear projection): the error $e = f_* - w^\top y$ of the best linear estimator must be uncorrelated with the data used in the estimator:

$$
\mathbb E[e, y] = 0 \quad\Rightarrow\quad \mathbb E[(f_* - w^\top y) y] = 0.
$$

Thus $\mathbb E[f_* y] - \mathbb E[y y^\top] w = 0$, i.e. $k_* - C_{yy} w = 0$, giving the same solution $w = C_{yy}^{-1} k_*$. So the GP predictor is the linear projection of $f_*$ onto the subspace spanned by the observed $y$.



## (Option 3) Joint Gaussian Conditioning

The standard GP route uses the joint distribution of $y$ and $f_*$ is Gaussian:


$$\begin{bmatrix} y \\ f_* \end{bmatrix}
\sim
\mathcal N\left(0,
\begin{bmatrix}
C_{yy} & k_* \\
k_*^\top & k_{**}
\end{bmatrix}
\right).$$

This can be done because the evaluation of a Gaussian Process at finite points is distributed as a normal. The conditional variable of a joint Gaussian is given by
$$
f_* | y \sim \mathcal N(k_*^\top C_{yy}^{-1} y , k_{**} - k_*^\top C_{yy}^{-1} k_*)
$$
Therefore, the conditional expectation and variance for a multivariate Gaussian yields
$$\mathbb E[f_* \mid y] = k_*^\top C_{yy}^{-1} y,
\qquad
\operatorname{Var}(f_* \mid y) = k_{**} - k_*^\top C_{yy}^{-1} k_*.$$


So the posterior mean equals the linear estimator we derived by minimizing MSE. For Gaussian processes, the posterior mean is both the Bayes (minimum mean squared error over all measurable estimators) and, as we've shown, the best linear estimator.

> *Important conceptual note: for a Gaussian prior the **optimal** estimator under squared loss is the conditional mean, and because of Gaussianity that conditional mean is linear in $y$. For non-Gaussian priors the best linear estimator (projection) is still $k_*^\top C_{yy}^{-1} y$ but it will not, in general, equal the conditional mean.*



## 1.2 Nonzero mean function

If the GP has mean function $m(\cdot)$, write $m_X = [m(x_1),\dots,m(x_n)]^\top$ and $m_* = m(x_*)$. The model is $y = m_X + (f - m_X) + \varepsilon$. Working with centered quantities $y - m_X$ and $f_* - m_*$ (which have zero mean) gives the estimator

$$
\mathbb E[f_* \mid y] = m_* + k_*^\top (K+\sigma^2 I)^{-1} (y - m_X).
$$


So you subtract the prior mean from observations, apply the same weight matrix, and add the prior mean back at the test point.


## 1.3 Complexity Analysis

- The training complexity is $\mathcal O(n^3)$ due to inversion of the matrix $C_{yy}$.
- The prediction
    - Mean: $\mathcal O(n)$ after precomputation of $C_{yy}^{-1}y$.
    - covariance: $\mathcal O(n^2)$ due to the multiplication of of $C_{yy}^{-1}k_*$.

But keep in mind that using GPUs this can be parallelized and improved drastically.

# 2. Practice and Code

For this implementation we will use the RBF Kernel:
$$
K(x,x') = e^{-\frac{1}{2\sigma}\|x-x'\|^2}
$$
This kernel is widely used because it is local and *universal* (Corollary 4.58, [[1]](#references)). This latter property it is very useful to guarante for asympotitacally proving that it converges to the best possible solution.

The implementation is straightforward using the previous section.
```python
def rbf_kernel(x1, x2, sigma=1.0):
    y = x1.reshape(-1, 1)-x2.reshape(1,-1)
    return np.exp(-0.5 / sigma**2 * np.square(y))

class GaussianProcess:
    def __init__(self, kernel=rbf_kernel, noise=1e-3):
        self.kernel = kernel
        self.noise = noise

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.K = self.kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        self.K_inv = np.linalg.inv(self.K)

    def predict(self, X_s):
        K_s = self.kernel(self.X_train, X_s)
        K_ss = self.kernel(X_s, X_s)

        mu_s = K_s.T.dot(self.K_inv).dot(self.y_train)
        cov_s = K_ss - K_s.T.dot(self.K_inv).dot(K_s)
        return mu_s, cov_s
```

Now we use a very simple dataset, fit the Gaussian Process and plot the results.

```python
# Training data (noisy observations)
X_train = np.array([[-4], [-3], [-2], [-1], [1]]).astype(float)
y_train = np.sin(X_train) + 0.1 * np.random.randn(*X_train.shape)

# Test points
X_test = np.linspace(-5, 5, 100).reshape(-1, 1)

# Fit GP and predict
gp = GaussianProcess()
gp.fit(X_train, y_train)
mu_s, cov_s = gp.predict(X_test)
std_s = np.sqrt(np.diag(cov_s))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'ro', label='Train data')
plt.plot(X_test, mu_s, 'b', label='Prediction mean')
plt.fill_between(X_test.ravel(),
mu_s.ravel() - 2 * std_s,
mu_s.ravel() + 2 * std_s,
alpha=0.2, label='Confidence interval')
plt.legend()
plt.title("Gaussian Process Regression")
plt.show()
```

![](gaussian_result.png)

One last aspect we have skipped during the implementation is that the matrix $C_{yy}$ is invertible. This is not a very good assumption and in practice it can lead to unstable behaviour. However, there is a nice patch using the Cholesky decomposition. 
$C_{yy}$ is positive definite since $C_{yy} = K + \sigma I$ where $K$ is positive semi-definite. Hence,
$$
C_{yy} = L L^\top
$$

**(1) Predictive mean:**
$$
\mathbb{E}[f_* \mid y] = k_*^\top C_{yy}^{-1} y
$$

Instead of computing $C_{yy}^{-1}$, we solve the equivalent linear system using the Cholesky factors:

1. Solve $L v = y$
2. Solve $L^\top \alpha = v$

Now $\alpha = C_{yy}^{-1} y$. So the mean is:
$$
\mathbb E[f_* \mid y] = k_*^\top \alpha
$$  


**(2) Predictive covariance**

The predictive covariance is:

$$
\operatorname{Var}(f_* \mid y) = k_{**} - k_*^\top C_{yy}^{-1} k_*
$$

Again, avoid $C_{yy}^{-1}$.
Instead compute:

1. Solve $L v = k_*$.
2. Then $k_*^\top C_{yy}^{-1} k_* = v^\top v$.

So:
$$
\operatorname{Var}(f_* \mid y) = k_{**} - v^\top v
$$

The result looks as follows

```python
class GaussianProcess:
    def __init__(self, kernel, noise=1e-3):
        self.kernel = kernel
        self.noise = noise

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        K = self.kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        self.L = np.linalg.cholesky(K)

    def predict(self, X_s):
        K_s = self.kernel(self.X_train, X_s)
        K_ss = self.kernel(X_s, X_s)

        v = np.linalg.solve(self.L, self.y_train) # First solve L v = y
        alpha = np.linalg.solve(self.L.T, v) # Then solve L^T alpha = v
        mu_s = K_s.T.dot(alpha)

        v = np.linalg.solve(self.L, K_s) # Solve for L\K_s
        cov_s = K_ss - v.T.dot(v)

        return mu_s, cov_s
```



# References

[1] Steinwart, I. and Christmann, A., 2008. Support vector machines. Springer Science & Business Media.  
[2] https://distill.pub/2019/visual-exploration-gaussian-processes/




# Appendix


## (A) Relation between Gaussian Process Regression and Maximum Mean Discrepancy (MMD)


Let $\mathcal H$ be the RKHS with reproducing kernel $k(\cdot,\cdot)$. For a location $x$ denote the representer $k_x := k(\cdot,x)\in\mathcal H$.
For two (signed) measures $\mu,\nu$ the squared MMD (kernel mean embedding distance) is
$$
\mathrm{MMD}^2(\mu,\nu) = \|\mu-\nu\|_{\mathcal H}^2
= \| m_\mu - m_\nu\|_{\mathcal H}^2,
$$
where $m_\mu=\int k(\cdot,x),d\mu(x)$ is the kernel mean map. For discrete/signed measures this reduces to kernel sums.

Take the target distribution $\mu=\delta_{x_*}$ (a unit mass at the test point) so $m_\mu = k_{x_*}$. Take a weighted empirical signed measure $\nu_w=\sum_{i=1}^n w_i \delta_{x_i}$ so $m_{\nu_w}=\sum_i w_i k_{x_i}$. Then
$$
\mathrm{MMD}^2\big(\delta_{x_*},\nu_w\big)
= \|k_{x_*} - \sum_{i} w_i k_{x_i}\|_{\mathcal H}^2.
$$

Use reproducing-kernel identities to expand that squared RKHS norm:
$$
\begin{aligned}
\|k_{x_*} - \sum_i w_i k_{x_i}\|_{\mathcal H}^2
&= \langle k_{x_*},k_{x_*}\rangle - 2\sum_i w_i\langle k_{x_*},k_{x_i}\rangle
\sum_{i,j} w_i w_j \langle k_{x_i},k_{x_j}\rangle \\
  &= k_{**} - 2 w^\top k_* + w^\top K w.
  \end{aligned}
  $$
  This is the MMD between $\delta_{x_*}$ and the weighted empirical measure $\nu_w$.


**Relation to the MSE quadratic form**

Recall from the linear-estimation derivation (zero-mean case) that the MSE of the linear estimator $\hat f_* = w^\top y$ equals
$$
\mathrm{MSE}(w)
= \mathbb E\big[(f_* - w^\top y)^2\big]
= k_{**} - 2 w^\top k_* + w^\top (K + \sigma^2 I) w.
$$

Compare this with the MMD expansion above: the difference is exactly the noise term $\sigma^2 w^\top w = \sigma^2 \|w\|_2^2$. Thus we can write
$$
\boxed{\mathrm{MSE}(w)
= \underbrace{\mathrm{MMD}^2 (\delta_{x_*},\nu_w )}_{\text{deterministic RKHS discrepancy}}+ \sigma^2 \|w\|_2^2.}
$$

So minimizing mean-squared error over linear estimators is *equivalent* to choosing weights $w$ that minimize the RKHS discrepancy (MMD) between the target at $x_*$ and the weighted training points, regularized by the squared $\ell_2$-norm of the weights weighted by the noise variance.
