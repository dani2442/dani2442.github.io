---
title: "The Linear Quadratic control problem in Infinite-dimensions"
date: 2026-03-28
tags: ["control theory", "PDEs"]
categories: ["pde", "optimal control"]
author: "Daniel López Montero"
showToc: true
draft: false
description: "Why the HJB is Bellman's equation in continuous time, why continuous time matters, and how to solve the resulting control problem with neural policy iteration."
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

% History: Linear Quadratic Regulator problem 1960s



Consider the following dynamical system
$$
\begin{align*}
\dot x &= Ax + Bu, \qquad x(0)=x_0\\
y &= Cx, \qquad 0\leq t\leq T
\end{align*}
$$

where $u\in L^2(0,T; U)$ and $y\in L^2(0,T; Y)$ and $U$ and $Y$ are Hilbert spaces. 


% Hypothesis, well-posedness



% Differential Riccati equation



## Examples


### Parabolic

Consider the equation 
$$
\dot x = Ax + Bu, \qquad y = Cx
$$
where $A$ is self-adjoint on a real Hilbert space. 
We assume that $A$ has a compact resolvent operator and that the spectrum of $A$ consists of a strictly decreasing sequence $\lambda_n, n\in \mathbb{N}$ of real eigenvalues with associated eigenvector $\phi_n\in H$ with $\|\phi_n\|=1$.

$$
\begin{align*}
z_t &= z_{\xi\xi}, \qquad 0<\xi<1\\
z_\xi(t,0) &= u(t), \quad z_\xi(t,1)=0,\ t>0\\
y(t) &= \int_0^1 c(\xi) z(t,\xi) d\xi 
\end{align*}
$$

### Hyperbolic

Consider the dynamical system
$$
\ddot z = Az + Bu, \qquad y = Cz
$$
where $A$ is self-adjoint on a real Hilbert space. 
We assume that $A$ has a compact resolvent operator and that the spectrum of $A$ consists of a strictly decreasing sequence $\lambda_n=-\omega_n^1, n\in \mathbb{N}$ satisfy $\omega_1\geq \delta$ and $\omega_{n+1} - \omega_{n}\geq\delta$ for all $n\in \mathbb{N}$.



$$
\begin{align*}
z_{tt} &= z_{\xi\xi}, \qquad 0<\xi<1\\
z_\xi(t,0) &= u(t), \quad z_\xi(t,1)=0,\ t>0\\
y(t) &= \int_0^1 c(\xi) z(t,\xi) d\xi 
\end{align*}
$$

### Delayed Differential Equations

Consider the following Retarded Differential Equation (RDF)

$$
\frac{d}{dt}(x(t) - Mx(t-\tau)) =  Lx(t-\tau) + Bu(t), \qquad y = Cx
$$
where $x(t)\in \mathbb{R}^n$, $u\in \mathbb{R}^m$, $M, L\in \mathbb{R}^{n\times n}$ and $B\in \mathbb{R}^{n\times m}$. 

## Bibliography