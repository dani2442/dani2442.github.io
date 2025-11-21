---
title: "Fusion Energy Simulation: Tokamak"
date: 2025-10-28
tags: ["machine learning", "control theory", "kernel methods"]
categories: ["control theory", "machine learning"]
author: "Daniel López Montero"
showToc: true
draft: true
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
The boundary conditions are as follows. All equations have a zero-derivative boundary condition at $\hat{\rho} = 0$. 
The $T_i$, $T_e$, $n_e$ equations have fixed boundary conditions at $\hat{\rho} = 1$, which are user-defined. 
The $\psi$ equation has a Neumann (derivative) boundary condition at $\hat{\rho} = 1$, 
which sets the total plasma current through the relation:

$$
I_p = 
\left[
\frac{\partial \psi}{\partial \rho} 
\frac{g_2 g_3}{\rho} 
\frac{R_0 J}{16 \pi^4 \mu_0}
\right]_{LCFS}
$$


## References

[1] Citrin, Jonathan, Ian Goodfellow, Akhil Raju, Jeremy Chen, Jonas Degrave, Craig Donner, Federico Felici et al. "TORAX: A fast and differentiable tokamak transport simulator in JAX." arXiv preprint arXiv:2406.06718 (2024).

[2] Felici, Federico. "Real-time control of tokamak plasmas: from control of physics to physics-based control." (2011).

[3] https://deepmind.google/discover/blog/bringing-ai-to-the-next-generation-of-fusion-energy/

## Appendix

### (A) Derivation of the 1D Transport Equations

> Derivation of the Poloidal Flux Diffusion Equation

We derive the poloidal flux diffusion equation, describing the temporal evolution of the poloidal flux under the assumption of static background flux surfaces. We follow [[2]](#references)
*Preliminaries*. We use the useful relation
$$
\begin{align*}
\langle \nabla \cdot \mathbf{F} \rangle &= \frac{\partial}{\partial V}\int (\nabla \cdot \mathbf{F}) dV = \frac{\partial}{\partial V} \int (\nabla \cdot \mathbf{F}) Rd\phi d\ell_p \frac{d\psi}{|\nabla\psi|} \\
&= \frac{\partial }{\partial V} \oint  \mathbf{F} \cdot \frac{\nabla V}{|\nabla V|} Rd\phi d\ell_p  \frac{d\psi}{|\nabla\psi|} \\
&= \frac{\partial }{\partial V}2\pi \oint \mathbf{F} \cdot \nabla V \frac{\partial \psi}{\partial V} \frac{Rd\ell_p}{|\nabla\psi|} \\
&= \frac{\partial}{\partial V} \langle \mathbf{F} \cdot \nabla V\rangle
\end{align*}
$$
Here, we used that the average of $\nabla \cdot \mathbf{F}$ over a flux surface can be represented by the derivative with respect to the enclosed volume $V$ of the volume integral of $\nabla \cdot \mathbf{F}$. Then we change to flux coordinates
$$
dV = Rd\phi d\ell_p \frac{d\psi}{|\nabla\psi|}
$$
where $\phi$ is the toroidal angle and $\ell_p$ the poloidal length along the flux surface.
Then we use Gauss divergence theorem to convert volume to surface integral,
$$
\int_V (\nabla \cdot \mathbf{F}) dV = \oint_{\partial V} \mathbf{F} \cdot \mathbf{n} dS = \oint_{\partial V} \mathbf{F} \cdot \frac{\nabla V}{|\nabla V|}dS
$$
Because $\phi$ is symmetric (axis) then the integral over $\phi$ gives a factor $2\pi$. Finally, the differential surface area of the flux surface is
$$dS = 2\pi R d\ell_p$$
Next, by the chain rule, we have
$$
|\nabla V| = \frac{\partial V}{\partial \psi} |\nabla \psi|
$$
Thus,
$$
\langle\mathbf{F} \cdot \nabla V\rangle = \int_V (\mathbf{F}\cdot \nabla V) dV = \oint \frac{(\mathbf{F}\cdot V) dS}{|\nabla V|} = \oint (\mathbf{F}\cdot \nabla \psi) \frac{\partial \psi}{\partial V} \frac{dS}{|\nabla \psi|}
$$
And this finishes the proof of the relation.


Consider a surface of constant poloidal flux whose boundary moves with velocity $\mathbf{u}_\psi$. For this surface:
$$
\frac{\partial \psi}{\partial t} + \mathbf{u}_\psi \cdot \nabla \psi = 0
$$

For a scalar field $F(t, \mathbf{x})$, define $H(t) = \int_V F\,dV$, where $V$ is the volume enclosed by $\psi = \text{const}$ moving with $\mathbf{u}_\psi$. By the Reynolds Transport Theorem, we have
$$
\begin{align*}
\frac{\partial H}{\partial t}\bigg|_{\psi=\text{const}} &=
\overbrace{\int_V \frac{\partial F}{\partial t} dV}^{\text{change inside volume}}+\overbrace{\oint_S F \mathbf{u}_\psi \cdot \mathbf{dS}_\psi}^{\text{change due to moving boundary}}\\
&=
\int_V \frac{\partial F}{\partial t} dV +
\oint_S F \mathbf{u}_\psi \cdot \frac{\nabla \psi}{|\nabla \psi|} dS
\end{align*}
$$

Using this equality, the time rate of change of toroidal flux $\Phi$ enclosed by $\psi = \text{const}$ is:
$$
\begin{aligned}
\frac{\partial \Phi}{\partial t}\bigg|_{\psi=\text{const}}
&= \frac{1}{2\pi} \frac{\partial}{\partial t} \int_V \mathbf{B}\cdot\nabla\phi\, dV \nonumber \\
&= \frac{1}{2\pi} \int_V \frac{\partial \mathbf{B}}{\partial t}\cdot\nabla\phi\, dV + \frac{1}{2\pi} \oint_S (\mathbf{B}\cdot\nabla\phi)(\mathbf{u}_\psi\cdot\nabla\psi) \frac{dS}{|\nabla\psi|}
\end{aligned}
$$

> Poloidal electric field.

Using Faraday’s law $\partial_t \mathbf{B} = -\nabla\times\mathbf{E}$:
$$
\begin{aligned}
\int_V \frac{\partial \mathbf{B}}{\partial t}\cdot\nabla\phi\, dV
&= -\int_V (\nabla\times\mathbf{E})\cdot\nabla\phi\, dV
= -\int_V \nabla\cdot(\mathbf{E}\times\nabla\phi)\, dV \nonumber\\
&= -\oint_S \mathbf{E}\cdot(\nabla\phi\times\nabla\psi)\, \frac{dS}{|\nabla\psi|}
= -2\pi \oint_S \mathbf{E}\cdot\mathbf{B}_p \frac{dS}{|\nabla\psi|}
\tag{C.5}
\end{aligned}
$$

> Toroidal electric field.

From Ampère’s law:
$$
\begin{aligned}
\nabla\psi\cdot\frac{\partial\mathbf{B}}{\partial t}
&= -\nabla\psi\cdot(\nabla\times\mathbf{E})
= \nabla\cdot(\nabla\psi\times\nabla\phi\, R E_\phi)
= -\nabla\cdot(2\pi \mathbf{B}_p R E_\phi)
\tag{C.6}
\end{aligned}
$$
Thus,
$$
\frac{\partial \psi}{\partial t} = 2\pi R E_\phi
\tag{C.7}
$$
and, using \eqref{C.2},
$$
\mathbf{u}_\psi \cdot \nabla\psi = -2\pi R E_\phi
\tag{C.8}
$$

Combining \eqref{C.5}–\eqref{C.8} gives:
$$
\begin{aligned}
\frac{\partial \Phi}{\partial t}\bigg|_{\psi=\text{const}}
&= -\oint_S (\mathbf{E}\cdot\mathbf{B}_p + B_\phi E_\phi) \frac{dS}{|\nabla\psi|}
= -\oint_S \mathbf{E}\cdot\mathbf{B} \frac{dS}{|\nabla\psi|}
= -\frac{\partial V}{\partial\psi}\langle \mathbf{E}\cdot\mathbf{B}\rangle
\tag{C.11}
\end{aligned}
$$

> Rate of change of poloidal flux.

$$
\begin{aligned}
\frac{\partial\psi}{\partial t}\bigg|_{\Phi=\text{const}}
&= \frac{\partial\psi}{\partial V}\frac{\partial V}{\partial\Phi}\frac{\partial\Phi}{\partial t}\bigg|_{\psi=\text{const}} \tag{C.12a}\\
\frac{\partial\psi}{\partial t}\bigg|_{\rho}
&+ \frac{\partial\psi}{\partial\rho}\frac{\partial\rho}{\partial t}\bigg|_{\Phi}= -\frac{\partial V}{\partial\Phi}\langle \mathbf{E}\cdot\mathbf{B}\rangle
\end{aligned}
$$

Using equilibrium relations, this becomes
$$
\frac{\partial\psi}{\partial t}\bigg|_{\rho} - \rho\frac{\dot{B}_0}{2B_0}\frac{\partial\psi}{\partial\rho}
= -2\pi R_0^2 \frac{\langle \mathbf{E}\cdot\mathbf{B}\rangle}{T\langle R_0^2/R^2\rangle}
$$

Define the equivalent cylindrical fields:
$$
\begin{aligned}
B_{p0} &= \frac{1}{2\pi R_0} \frac{\partial\psi}{\partial\rho} \tag{C.13}\\
E_0 &= R_0 \frac{\langle \mathbf{E}\cdot\mathbf{B}\rangle}{T\langle R_0^2/R^2\rangle}\tag{C.14}
\end{aligned}
$$
Then, for $\dot{B}_0=0$:
$$
\frac{\partial B_{p0}}{\partial t} = -\frac{\partial E_0}{\partial\rho}
\tag{C.15}
$$

> Ohm’s law.

$$
\langle \mathbf{j}\cdot\mathbf{B}\rangle = \sigma_\parallel \langle \mathbf{E}\cdot\mathbf{B}\rangle + \langle \mathbf{j}_{ni}\cdot\mathbf{B}\rangle
\tag{C.16}
$$
where $\mathbf{j}_{ni} = \mathbf{j}_{bs} + \mathbf{j}_{cd}$.

Thus,
$$
\sigma_\parallel E_\parallel = j_\parallel - j_{bs} - j_{cd}
\tag{C.18}
$$

> Parallel current.

$$
\begin{aligned}
\frac{\langle \mathbf{j}\cdot\mathbf{B}\rangle}{B_0}
&= \frac{2\pi R_0 J^2}{\mu_0 V'} \frac{\partial}{\partial\rho}
\left(\frac{G^2}{J}\frac{\partial\psi}{\partial\rho}\right)
\tag{C.19}
\end{aligned}
$$
with
$$
J = \frac{T}{R_0 B_0}, \quad
G^2 = \frac{V'}{4\pi^2}\left\langle \frac{(\nabla\rho)^2}{R^2} \right\rangle, \quad
V' = \frac{\partial V}{\partial\rho}
\tag{C.20}
$$

> The poloidal flux diffusion equation.
Combining (C.17), (C.19), and (C.12c):
$$
\sigma_\parallel \left(
\frac{\partial\psi}{\partial t} + \rho\frac{\dot{B}_0}{2B_0}\frac{\partial\psi}{\partial\rho}
\right)
= \frac{R_0 J^2}{\mu_0 \rho}\frac{\partial}{\partial\rho}
\left(\frac{G^2}{J}\frac{\partial\psi}{\partial\rho}\right)- \frac{V'}{2\pi\rho}(j_{bs} + j_{cd})
\tag{C.21}
$$

> Derivation of the Particle Transport Equation

For a species $\alpha$:
$$
\frac{\partial n_\alpha}{\partial t} + \nabla\cdot(n_\alpha \mathbf{u}_\alpha) = s_\alpha
\tag{C.22}
$$

Integrating over the volume enclosed by a flux surface:
$$
\int \frac{\partial n_\alpha}{\partial t} dV +  \oint n_\alpha \mathbf{u}_\alpha\cdot\frac{\nabla\Phi}{|\nabla\Phi|} dS = \int s_\alpha dV
\tag{C.23}
$$

Using Eq.~(C.3):
$$
\frac{\partial}{\partial t}\bigg|_\Phi \int n_\alpha dV + \oint n_\alpha (\mathbf{u}_\alpha - \mathbf{u}_\Phi)\cdot\frac{\nabla\rho}{|\nabla\rho|} dS
= \int s_\alpha dV
\tag{C.24}
$$

Using flux surface averaging:
$$
\frac{\partial}{\partial t}\bigg|_\Phi \left[ \langle n_\alpha\rangle V' \right] + \frac{\partial}{\partial\rho}\left[ V'\langle n_\alpha(\mathbf{u}_\alpha - \mathbf{u}_\Phi)\cdot\nabla\rho\rangle \right]
= \langle s_\alpha\rangle V'
\tag{C.26}
$$

Finally, at constant $\rho$:
$$
\frac{1}{V'}\left(\frac{\partial}{\partial t} + \frac{\dot{B}_0}{2B_0}\rho\frac{\partial}{\partial\rho}\right)
(\langle n_\alpha\rangle V') + \frac{1}{V'}\frac{\partial\Gamma_\alpha}{\partial\rho}
= S_\alpha
\tag{C.27}
$$
where $\Gamma_\alpha$ is the particle flux and $S_\alpha$ the source term.
