---
title: "AI Psychosis"
date: 2026-01-20
tags: ["oversupply", "economics", "creativity", "AI"]
categories: ["miscellaneous", "machine learning", "philosophy"]
author: "Daniel López Montero"
showToc: true
draft: true
description: "Losing touch with reality in the age of AI."
ShowWordCount: false
ShowReadingTime: true
comments: true
UseHugoToc: true
editPost:
    URL: "https://github.com/dani2442/dani2442.github.io/tree/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

What is AI psychosis? and how to avoid it?

Last week, a guy on Twitter claimed that he solved one of the Millennium Prize Problems (Navier-Stokes). This guy is ex-director of DeepMind, so a lot of people took him seriously despite not having formal mathematical background. He even made a 10,000$ bet that he was right, however, in the end, he offered an *AI-slop* proof.

They coined the term *AI psychosis* to describe this state of mind: disconnect between our perception of reality and the actual state of the world, fueled by the overwhelming presence of AI-generated content. 

> We may ask ourselves: what went wrong here?

We all know that LLMs tend to agree with us due to the way they are post-trained using human feedback and RL. What I want to discuss is not self-delusion, but rather a more subtle effect: the collapse of thought and lost contact with reality. 

One thing we know for sure is that when humans becomes isolated from society, they may develop psychosis or any other mental disorder, without external stimili. However, this is not a purely human phenomenon, LLMs can experience a similar effect. Indeed, if we consider the continuous version of self-attention.


Consider the dissipative self-attention equation:
$$
\dot X(t) = -\gamma X(t) + \text{Attn}(X(t)), \quad \text{where}\quad \text{Attn}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V W_O,
$$
where $Q = XW_Q$, $K = XW_K$, and $V = XW_V$. Then, we can state the following theorem:

> **Theorem 3 (Exponential convergence under contraction)**
Assume $\mathrm{Attn}$ is Lipschitz on the region visited by trajectories with constant $L$.
If $\gamma>L$, then:
> 1. the ODE has a unique equilibrium $X_\star$,
> 2. every solution converges exponentially:
>   $$
   \|X(t)-X_\star\|\le e^{-(\gamma-L)t}\|X(0)-X_\star\|.
   $$

*Proof:* For two solutions $X,Y$, we have
$$
\frac{d}{dt}\|X-Y\|
\le \|-\gamma(X-Y)+(\mathrm{Attn}(X)-\mathrm{Attn}(Y))\|
\le (-\gamma+L)\|X-Y\|.
$$
Grönwall gives $\|X(t)-Y(t)\|\le e^{-(\gamma-L)t}\|X(0)-Y(0)\|$, i.e. contraction of the flow.
A contracting flow has a unique fixed point, and all trajectories converge to it.