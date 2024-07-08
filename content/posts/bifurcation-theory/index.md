---
title: "Nonlinear Functional Analysis: Bifurcation Theory"
date: 2024-07-07
tags: ["mathematics"]
author: "Daniel López Montero"
showToc: true
draft: true
description: "Introduction to Uniparametric Bifurcation Theory."
ShowWordCount: false
ShowReadingTime: true
comments: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/dani2442/dani2442.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---


Several non-linear problems relevant in practical applications can be expressed as a fixed point equation. In many cases, it is crucial to investigate how the model’s behavior changes with variations in a parameter, denoted as $\lambda$. In practical applications, $\lambda$ represents a physical or empirical magnitude of interest. Bifurcation Theory is a subfield in Nonlinear Functional Analysis that tries to study the general behavior of the equations that can be written as $\mathfrak{F}(\lambda, u)=0$ where $\lambda$ is the *bifurcation parameter*.

## Examples of Bifurcations

### One-Dimensional Example
Considering the one-dimensional problem where $U=\mathbb{R}$
$$
    \mathfrak{F}(\lambda, u) = \lambda u + \frac{1}{2}\cos(2\lambda)u^2
$$
Here, $\mathfrak{F}$ exhibits a bifurcation behavior at $(\lambda, u)=(0,0)$ as shown in the following figure.
![Example 1](surface.svg)
Here, we have a trivial branch of solutions $(\lambda, u)=(\lambda, 0)$ for every $\lambda\in\mathbb{R}$.

### Logistic Map
Another example of bifurcation emerges from the logistic map, 
$$
x_{n+1} = rx_n(1-x_n)
$$
The chaotic behavior of arises from a very simple nonlinear dynamical equation. And it describes the many physical effects such as the population reproduction and starvation of a population, i.e., given a population $x\in [0,1]$ where 1 represent the population maximum capacity, the population of the next year can be given by $x_{n+1} = rx_n$. However, we need to further constraint to the maximum capacity, and that is why the factor $(x_n-1)$ appears. This equation exhibits a bifurcation behavior in the stable points (the limit of the sequences) where we take the bifurcation parameter $r$. You can see in the next figure that multiple bifurcation points emerge (the most notorious is when $r=2$)
![Example 2](logistic_map.png)

## Bifurcation from Simple Eigenvalues

One of the most important theorems in Bifurcation Theorey is the Crandall-Rabinowitz Theorem [1] which state the sufficient conditions to find a simple bifurcation point, it is build upon the Implicit Function Theorem and therefore the convergence of the Newton method. Firstly, it defines a function $\mathfrak{G}$ as follows

\[\tag{1}
\mathfrak{G}(s, \lambda, y):=
\begin{cases}
s^{-1}\mathfrak{F}(\lambda, s(\varphi_0 + y)) & \text{if }s\not=0\\
D_u\mathfrak{F}(\lambda, 0)(\varphi_0 + y)& \text{if }s=0
\end{cases}
\]
where $\varphi_0$ is the simple eigenvector of  $D_u \mathfrak{F}_u(\lambda_0)$ whose corresponding eigenvalue is 0. Using the Implicit Function Theorem, we can prove that there exists two unique mappings $\lambda(s)$ and $y(s)$ such that $\mathfrak{G}(s, \lambda(s), y(s))=0$.

### Application to equation: $-u'' = \lambda u - a(x) u^3$

We will study the general case of the equation
$$\tag{2}
    \begin{cases}
        -u'' = \lambda u - a(x)u^3\quad\quad \text{in}\quad (0,L),\\
        u(0) = u(L) = 0
    \end{cases}
$$
where $a\in\mathcal{C}[0,L]$. Using the Crandall-Rabinowitz Theorem, we can prove that there are bifurcation points at $(\lambda, u) = (\frac{n^2\pi^2}{L^2}, 0)$ for every $n\geq 1$. And the bifurcation diagram will look like the following figure.
![Application](bifurcation_diagram_global.png)
We will introduce in the following section some numerical methods for bifurcation theory.

## Numerical Methods in Bifurcation Theory

