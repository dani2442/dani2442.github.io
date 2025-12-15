---
title: "Fusion Energy Simulation: Tokamak"
date: 2025-10-28
tags: ["machine learning", "control theory", "kernel methods"]
categories: ["control theory", "machine learning"]
author: "Daniel López Montero"
showToc: true
draft: true
description: "A short tour of TORAX’s 1D tokamak transport PDEs."
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

![alt text](tokamak.png)
*Source: [[2]](#references)*


TORAX solves coupled 1D PDEs in normalized toroidal flux coordinates, $\hat \rho$, with $0 \leq \hat \rho \leq 1.$

![alt text](equations.png)

| Symbol | Description | Units / Notes |
|:-------|:-------------|:--------------|
| $ {\color{red}{T_i}}, {\color{orange}{T_e}} $ | Ion and electron temperatures | keV or eV |
| $ n_i, {\color{blue}{n_e}} $ | Ion and electron densities | m⁻³ |
| $ {\color{teal}{\psi}} $ | Poloidal magnetic flux | Wb (Weber) |
| $ \Phi(\psi) $ | Toroidal magnetic flux enclosed by the magnetic poloidal flux surface | Wb |
| $ \Phi_b $ | Toroidal flux enclosed by the plasma boundary (last-closed-flux surface) | Wb |
| $ \hat{\rho}= \sqrt{\tfrac{\Phi({\color{teal}{\psi}})}{\Phi_b}} $ | Normalized toroidal flux coordinate | Dimensionless |
| $ V'\equiv\tfrac{dV}{d\hat \rho} $ | Derivative of plasma volume w.r.t. $ \hat{\rho} $, i.e. flux surface volume derivative | m³ |
| $ g_0 = \langle \nabla V \rangle $ | Flux-surface-averaged gradient of the plasma volume | m² |
| $ g_1 = \langle (\nabla V)^2 \rangle $ | Flux-surface-averaged square of the volume gradient | m⁴ |
| $ g_2 = \left\langle \frac{(\nabla V)^2}{R^2} \right\rangle $ | Flux-surface-averaged geometric factor involving major radius | m² |
| $ g_3 = \left\langle \frac{1}{R^2} \right\rangle $ | Flux-surface-averaged reciprocal of squared major radius | 1/m² |
| $ R $ | Major radius along the flux surface | m |
| $ B_\varphi $ | Toroidal magnetic field | T (Tesla) |
| $ \mathbf{B} $ | Total magnetic field | T (Tesla) |
| $ \sigma_\parallel $ |  Plasma neoclassical conductivity | S/m |
| $ \mu_0 $ | Magnetic permeability of free space | $4\pi \times 10^{-7}$ H/m |
| $ F = R B_\varphi $ | Toroidal field flux function | T·m |
| $ \chi_i, \chi_e $ | Ion and electron heat conductivities | m²/s |
| $ q_{0i}^{conv}, q_{0e}^{conv} $ | Ion/electron heat convective losses | W/m³ |
| $ D_e $ | Electron particle diffusivity | m²/s |
| $ V_e $ | Electron particle convection velocity | m/s |
| $ Q_i, Q_e $ | Ion and electron heat sources | W/m³ |
| $ S_n $ | Total electron particle source | 1/s or m⁻³·s⁻¹ |
| $ \mathbf{j}_{ni} $ | Non-inductive current density (bootstrap + external drive) | A/m² |
| $ I_p $ | Total plasma current | A |
| $ J $ | Total toroidal plasma current density | A/m² |
| $ R_0 $ | Major radius at magnetic axis | m |
| $ \langle \cdot \rangle $ | Flux-surface average | — |

**Boundary Conditions:**
The boundary conditions are as follows. All equations have a zero-derivative (Neumann) boundary condition at $\hat{\rho} = 0$. 
The $T_i$, $T_e$, $n_e$ equations have fixed boundary conditions at $\hat{\rho} = 1$, which are user-defined. 
The $\psi$ equation has a Neumann (derivative) boundary condition at $\hat{\rho} = 1$, 
which sets the total plasma current through the relation:

$$
I_p = 
\left[
\frac{\partial \psi}{\partial \hat{\rho}} 
\frac{g_2 g_3}{\hat{\rho}} 
\frac{R_0 J}{16 \pi^4 \mu_0}
\right]_{LCFS}
$$

The equations are derived from the transport equations in toroidal geometry under the assumption of static background flux surfaces.



## Ion and Electron Heat Transport Equations

TORAX evolves the ion and electron temperatures via the 1D heat transport PDEs (see [[1]](#references)):


$$
\boxed{\;
\begin{aligned}
\frac{3}{2} V'^{-5/3}
\left( 
\frac{\partial}{\partial t}
- \frac{\dot{\Phi}_b}{2\Phi_b} 
\hat{\rho}\frac{\partial}{\partial \hat{\rho}}
\right)
\left[ 
V'^{5/3} n_i {{T_i}}
\right]
&=
\frac{1}{V'} 
\frac{\partial}{\partial \hat{\rho}}
\left[
\chi_i n_i 
\frac{g_1}{V'} 
\frac{\partial {T_i}}{\partial \hat{\rho}}
- g_0 q_i^{\text{conv}} {T_i}
\right]
+ Q_i \\
\frac{3}{2} V'^{-5/3}
\left( 
\frac{\partial}{\partial t}
- \frac{\dot{\Phi}_b}{2\Phi_b} 
\hat{\rho}\frac{\partial}{\partial \hat{\rho}}
\right)
\left[ 
V'^{5/3} {n_e} {T_e}
\right]
&=
\frac{1}{V'} 
\frac{\partial}{\partial \hat{\rho}}
\left[
\chi_e {n_e} 
\frac{g_1}{V'} 
\frac{\partial {T_e}}{\partial \hat{\rho}}
- g_0 q_e^{\text{conv}} {T_e}
\right]
+ Q_e 
\end{aligned}\;}
$$

## Electron Particle Transport Equation

TORAX evolves the electron density via the 1D particle transport PDE (see [[1]](#references)):
$$
\boxed{\;
\left(\frac{\partial}{\partial t}-\frac{\dot\Phi_b}{2\Phi_b}\hat\rho\frac{\partial}{\partial \hat\rho}\right)(n_eV')
=
\frac{\partial}{\partial \hat\rho}\Big[
D_e n_e \frac{g_1}{V'} \frac{\partial n_e}{\partial \hat\rho}
-
g_0 V_e n_e
\Big]
+V'S_n.\;}
$$

A detailed derivation (flux-surface averaging + moving-grid term from $\dot\Phi_b$) is sketched in the [Appendix](#appendix).


## Current Diffusion Equation 
TORAX evolves the poloidal flux via the 1D current diffusion PDE (see [[1]](#references)):
$$
\boxed{\;
\frac{16 \pi^2 \sigma_{\parallel}\mu_0 \hat\rho\, \Phi_b^2}{F^2}
\left(
\frac{\partial \psi}{\partial t}
- \frac{\hat\rho\, \dot{\Phi}_b}{2 \Phi_b} 
\frac{\partial \psi}{\partial \hat\rho}
\right)
=
\frac{\partial}{\partial \hat\rho}
\left(
\frac{g_2 g_3}{\hat\rho} 
\frac{\partial \psi}{\partial \hat\rho}
\right)
- \frac{8 \pi^2 V' \mu_0 \Phi_b}{F^2}
\langle \mathbf{B} \cdot \mathbf{j}_{ni} \rangle\;}
$$

A more detailed derivation of the underlying flux-surface-averaged form is given in [[2]](#references) (and sketched in the [Appendix](#appendix)).
## References

[1] Citrin, Jonathan, Ian Goodfellow, Akhil Raju, Jeremy Chen, Jonas Degrave, Craig Donner, Federico Felici et al. "TORAX: A fast and differentiable tokamak transport simulator in JAX." arXiv preprint arXiv:2406.06718 (2024).

[2] Felici, Federico. "Real-time control of tokamak plasmas: from control of physics to physics-based control." (2011).

[3] https://deepmind.google/discover/blog/bringing-ai-to-the-next-generation-of-fusion-energy/

[4] Hinton, F. L. and R. D. Hazeltine (1976). “Theory of plasma transport in toroidal confinement systems.” In: Rev. Mod. Phys. 48.2, pp. 239–308. doi: 10.1103/RevModPhys. 48.239.

## Appendix

### (A) Derivation of the 1D Transport Equations

> Derivation of the Poloidal Flux Diffusion Equation

We give a compact sketch of the main identities and coordinate changes used by TORAX; for a full derivation of the current diffusion equation see [[2]](#references).

#### A.1 Flux-Surface Averaging Identities

Let $\psi$ label nested flux surfaces and $V(\psi)$ be the volume enclosed by a given surface. Define the flux-surface average of a scalar $A$ by
$$
\langle A\rangle(\psi)
\;\equiv\;
\frac{1}{V_\psi'}\oint_{\psi=\mathrm{const}}\frac{A}{|\nabla\psi|}\,dS,
\qquad
V_\psi' \equiv \frac{\partial V}{\partial \psi}
=\oint_{\psi=\mathrm{const}}\frac{1}{|\nabla\psi|}\,dS. \tag{A.0}
$$
Applying the divergence theorem to the thin shell between $\psi$ and $\psi+d\psi$ yields, for any vector field $\mathbf{F}$,
$$
\left\langle \nabla\cdot \mathbf{F}\right\rangle
= \frac{1}{V_\psi'}\frac{\partial}{\partial\psi}\left(V_\psi'\left\langle \mathbf{F}\cdot\nabla\psi\right\rangle\right). \tag{A.0b}
$$

#### A.2 Moving-Grid Term From $\dot\Phi_b$

TORAX uses $\hat\rho=\sqrt{\Phi(\psi)/\Phi_b(t)}$. Holding the physical toroidal flux label $\Phi$ fixed implies
$$
0=\left.\frac{\partial}{\partial t}\right|_{\Phi}\left(\hat\rho^2\Phi_b\right)
= 2\hat\rho\,\Phi_b\left.\frac{\partial \hat\rho}{\partial t}\right|_{\Phi}+\hat\rho^2\dot\Phi_b,
$$
so
$$
\left.\frac{\partial \hat\rho}{\partial t}\right|_{\Phi}
=-\frac{\dot\Phi_b}{2\Phi_b}\hat\rho. \tag{A.0c}
$$
For any profile $f(\hat\rho,t)$, the chain rule gives the operator used throughout TORAX:
$$
\left.\frac{\partial f}{\partial t}\right|_{\Phi}
=
\left.\frac{\partial f}{\partial t}\right|_{\hat\rho}
-\frac{\dot\Phi_b}{2\Phi_b}\hat\rho\frac{\partial f}{\partial \hat\rho}. \tag{A.0d}
$$

Consider a surface of constant poloidal flux whose boundary moves with velocity $\mathbf{u}_\psi$. For this surface:
$$
\frac{\partial \psi}{\partial t} + \mathbf{u}_\psi \cdot \nabla \psi = 0 \tag{A.1}
$$

For a scalar field $F(t, \mathbf{x})$, define $H(t) = \int_V F\,dV$, where $V$ is the volume enclosed by $\psi = \text{const}$ moving with $\mathbf{u}_\psi$. By the Reynolds Transport Theorem, we have
$$
\begin{align*}
\frac{\partial H}{\partial t}\bigg|_{\psi=\text{const}} &=
\overbrace{\int_V \frac{\partial F}{\partial t} dV}^{\text{change inside volume}}+\overbrace{\oint_S F\, \mathbf{u}_\psi \cdot \mathbf{n}\, dS}^{\text{change due to moving boundary}}\\
&=
\int_V \frac{\partial F}{\partial t} dV +
\oint_S F \mathbf{u}_\psi \cdot \frac{\nabla \psi}{|\nabla \psi|} dS
\end{align*}\tag{A.1b}
$$

Using this equality, the time rate of change of toroidal flux $\Phi$ enclosed by $\psi = \text{const}$, i.e., $F=\mathbf{B}\cdot\nabla\phi$ and $\Phi(t)=\int_V \mathbf{B}\cdot\nabla\phi\, dV$, is:
$$
\begin{aligned}
\frac{\partial \Phi}{\partial t}\bigg|_{\psi=\text{const}}
&= \frac{1}{2\pi} \frac{\partial}{\partial t} \int_V \mathbf{B}\cdot\nabla\phi\, dV \nonumber \\
&= \frac{1}{2\pi} \int_V \frac{\partial \mathbf{B}}{\partial t}\cdot\nabla\phi\, dV + \frac{1}{2\pi} \oint_S (\mathbf{B}\cdot\nabla\phi)(\mathbf{u}_\psi\cdot\nabla\psi) \frac{dS}{|\nabla\psi|}
\end{aligned}\tag{A.2}
$$

> 1. Poloidal electric field.

Using Faraday's law $\partial_t \mathbf{B} = -\nabla\times\mathbf{E}$, we can rewrite the first volume integral (A.2) as:
$$
\begin{aligned}
\int_V \frac{\partial \mathbf{B}}{\partial t}\cdot\nabla\phi\, dV
&= -\int_V (\nabla\times\mathbf{E})\cdot\nabla\phi\, dV
= -\int_V \nabla\cdot(\mathbf{E}\times\nabla\phi)\, dV \nonumber\\
&= -\oint_S (\mathbf{E}\times \nabla \phi)\frac{\nabla\psi}{|\nabla \psi|}dS \tag{Gauss}\\
&= -\oint_S \mathbf{E}\cdot(\nabla\phi\times\nabla\psi)\, \frac{dS}{|\nabla\psi|}
= -2\pi \oint_S \mathbf{E}\cdot\mathbf{B}_p \frac{dS}{|\nabla\psi|}
\end{aligned}
$$
where $\mathbf{B}_p \equiv \frac{1}{2\pi}\nabla\phi\times\nabla\psi$ is the poloidal magnetic field.

> 2. Toroidal electric field.

The boundary-motion term in (A.2) can be expressed in terms of the toroidal electric field. Following [[2]](#references), one uses the kinematic relation
$$
\mathbf{u}_\psi\cdot\nabla\psi = -2\pi R E_\phi, \tag{A.2b}
$$
together with $\mathbf{B}\cdot\nabla\phi = B_\phi/R$, to obtain
$$
\frac{1}{2\pi}\oint_S (\mathbf{B}\cdot\nabla\phi)(\mathbf{u}_\psi\cdot\nabla\psi)\frac{dS}{|\nabla\psi|}
= -\oint_S B_\phi E_\phi\,\frac{dS}{|\nabla\psi|}.
$$
Combining this with step 1 yields
$$
\begin{aligned}
\frac{\partial \Phi}{\partial t}\bigg|_{\psi=\text{const}}
&= -\oint_S (\mathbf{E}\cdot\mathbf{B}_p + B_\phi E_\phi)\,\frac{dS}{|\nabla\psi|}
= -\oint_S \mathbf{E}\cdot\mathbf{B}\,\frac{dS}{|\nabla\psi|}
= -\frac{\partial V}{\partial\psi}\langle \mathbf{E}\cdot\mathbf{B}\rangle.
\end{aligned}\tag{A.2c}
$$

> 3. Closing the 1D current diffusion equation.

Equation (A.2c) relates the evolution of toroidal flux to the flux-surface-averaged parallel electric field via $\langle \mathbf{E}\cdot\mathbf{B}\rangle$. To obtain a closed 1D PDE for $\psi$ (and the normalized form used by TORAX), one combines:
- A parallel Ohm's law closure, which relates $\langle \mathbf{E}\cdot\mathbf{B}\rangle$ to $j_\parallel$ and non-inductive current drive via $\sigma_\parallel$.
- Ampere's law + flux-surface averaging, which expresses $j_\parallel$ as a radial diffusion operator acting on $\psi$, with geometry packaged into $g_2$ and $g_3$.
- The moving-grid identity (A.0d), which converts time derivatives at fixed toroidal flux label $\Phi$ into derivatives at fixed $\hat\rho$ when $\Phi_b(t)$ changes.

The resulting normalized form is the TORAX current diffusion equation shown in the main text.

> Derivation of the Particle Transport Equation

For an arbitrary plasma species $\alpha$, let $n_\alpha$ be the density and $s_\alpha$ a particle source. Start from the continuity equation:
$$
\frac{\partial n_\alpha}{\partial t}+\nabla\cdot(n_\alpha \mathbf{u}_\alpha)=s_\alpha.
$$
Flux-surface averaging (using the divergence identity (A.0b) and writing $V'=dV/d\hat\rho$) yields the 1D conservation law at fixed toroidal-flux label $\Phi$:
$$
\left.\frac{\partial}{\partial t}\right|_{\Phi}\left(n_\alpha V'\right)
= -\frac{\partial \Gamma_\alpha}{\partial \hat\rho}+V'S_\alpha,
$$
where $S_\alpha\equiv\langle s_\alpha\rangle$ and $\Gamma_\alpha$ is the flux-surface-averaged radial particle flux (the term generated by $\langle\nabla\cdot(n_\alpha\mathbf{u}_\alpha)\rangle$ via (A.0b)).
Finally, converting the time derivative from fixed $\Phi$ to fixed $\hat\rho$ using (A.0d) gives the TORAX moving-grid operator:
$$
\left(\frac{\partial}{\partial t}-\frac{\dot\Phi_b}{2\Phi_b}\hat\rho\frac{\partial}{\partial \hat\rho}\right)\left(n_\alpha V'\right)
= -\frac{\partial \Gamma_\alpha}{\partial \hat\rho}+V'S_\alpha.
$$
