"""
SO(3) Group Convolution (Peter–Weyl / Wigner D-matrices)
========================================================

This script demonstrates *left-equivariant* group convolution on SO(3):

  (f ⋆ L)(g) = ∫_{SO(3)} f(g h⁻¹) L(h) dμ(h)

with normalized Haar measure dμ. Using ZYZ Euler angles (α, β, γ):

  dμ = (1 / 8π²) sin(β) dα dβ dγ

We compute Fourier (Peter–Weyl) coefficients using Wigner D-matrices:

  f̂^ℓ = ∫ f(g) D^ℓ(g)† dμ(g)   (a (2ℓ+1)×(2ℓ+1) matrix)

and the convolution theorem for the convention above is:

  (f ⋆ L)̂^ℓ = L̂^ℓ  f̂^ℓ    (note the order!)

Everything is computed on a tensor-product Euler grid and truncated to ℓ ≤ L_MAX,
so results are approximate unless the sampled functions are band-limited.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

# =========================================================
# Configuration
# =========================================================

N_ALPHA = 16
N_BETA = 8
N_GAMMA = 16

L_MAX = 4

# A fixed left-translation used for the equivariance check.
SHIFT_EULER_ZYZ = (np.pi / 3, np.pi / 4, np.pi / 6)


# =========================================================
# Wigner D-matrices (ZYZ convention)
# =========================================================

def wigner_d_small(l: int, beta: float) -> np.ndarray:
    """
    Compute the small Wigner d-matrix d^l_{m,n}(β) for all m,n in [-l, l].
    Uses the standard closed-form summation formula (suitable for small ℓ).
    Returns a (2l+1, 2l+1) matrix.
    """
    d = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.float64)
    
    c = np.cos(beta / 2)
    s = np.sin(beta / 2)
    
    for m in range(-l, l + 1):
        for n in range(-l, l + 1):
            # Using the general formula for Wigner d-matrix
            s_min = max(0, n - m)
            s_max = min(l + n, l - m)
            
            val = 0.0
            for ss in range(s_min, s_max + 1):
                # Use log-factorials (via lgamma) to avoid huge Python ints that
                # break `np.sqrt` for n >= 21 and to improve numeric stability.
                log_num = 0.5 * (
                    math.lgamma(l + n + 1)
                    + math.lgamma(l - n + 1)
                    + math.lgamma(l + m + 1)
                    + math.lgamma(l - m + 1)
                )
                log_den = (
                    math.lgamma(l + n - ss + 1)
                    + math.lgamma(ss + 1)
                    + math.lgamma(l - m - ss + 1)
                    + math.lgamma(ss + m - n + 1)
                )
                
                power_c = 2 * l + n - m - 2 * ss
                power_s = 2 * ss + m - n
                
                sign = (-1) ** (ss + m - n)
                val += sign * math.exp(log_num - log_den) * (c ** power_c) * (s ** power_s)
            
            d[m + l, n + l] = val
    
    return d


def wigner_D_matrix(l: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Compute the Wigner D-matrix D^l_{m,n}(α, β, γ) for all m,n in [-l, l].
    
    D^l_{m,n}(α, β, γ) = e^{-i m α} d^l_{m,n}(β) e^{-i n γ}
    
    Returns a (2l+1, 2l+1) complex matrix.
    """
    d = wigner_d_small(l, beta)
    
    m_vals = np.arange(-l, l + 1)
    n_vals = np.arange(-l, l + 1)
    
    phase_alpha = np.exp(-1j * m_vals * alpha)
    phase_gamma = np.exp(-1j * n_vals * gamma)
    
    D = phase_alpha[:, np.newaxis] * d * phase_gamma[np.newaxis, :]
    
    return D


# =========================================================
# Grid + quadrature weights (midpoint rule in β to avoid poles)
# =========================================================

def make_euler_zyz_grid(
    n_alpha: int, n_beta: int, n_gamma: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.linspace(0.0, 2 * np.pi, n_alpha, endpoint=False)
    beta = (np.arange(n_beta) + 0.5) * (np.pi / n_beta)
    gamma = np.linspace(0.0, 2 * np.pi, n_gamma, endpoint=False)

    d_alpha = 2 * np.pi / n_alpha
    d_beta = np.pi / n_beta
    d_gamma = 2 * np.pi / n_gamma

    Alpha, Beta, Gamma = np.meshgrid(alpha, beta, gamma, indexing="ij")
    weights = (np.sin(Beta) / (8 * np.pi**2)) * (d_alpha * d_beta * d_gamma)

    angles = np.stack([Alpha.ravel(), Beta.ravel(), Gamma.ravel()], axis=1)
    rotations = Rotation.from_euler("ZYZ", angles).as_matrix()
    rotations = rotations.reshape(n_alpha, n_beta, n_gamma, 3, 3)

    return alpha, beta, gamma, weights, rotations


def precompute_wigner_D_grid(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, l_max: int
) -> dict[int, np.ndarray]:
    """
    Precompute D^l(α_i, β_j, γ_k) on the Euler tensor grid for all l ≤ l_max.

    Returns a dict: l -> array of shape (Nα, Nβ, Nγ, 2l+1, 2l+1).
    """
    D_grid: dict[int, np.ndarray] = {}
    for l in range(l_max + 1):
        m_vals = np.arange(-l, l + 1)
        n_vals = np.arange(-l, l + 1)

        phase_alpha = np.exp(-1j * alpha[:, None] * m_vals[None, :])  # (Nα, 2l+1)
        phase_gamma = np.exp(-1j * gamma[:, None] * n_vals[None, :])  # (Nγ, 2l+1)

        d_beta = np.stack([wigner_d_small(l, b) for b in beta], axis=0)  # (Nβ, 2l+1, 2l+1)

        D = (
            phase_alpha[:, None, None, :, None]
            * d_beta[None, :, None, :, :]
            * phase_gamma[None, None, :, None, :]
        )
        D_grid[l] = D

    return D_grid


def fourier_coefficients_from_samples(
    samples: np.ndarray, D_grid: dict[int, np.ndarray], weights: np.ndarray
) -> dict[int, np.ndarray]:
    """
    Compute Fourier coefficients for f given samples f(α_i, β_j, γ_k).

      f̂^l = ∫ f(g) D^l(g)† dμ(g)
    """
    coeffs: dict[int, np.ndarray] = {}
    for l, D in D_grid.items():
        D_dag = np.conj(D).swapaxes(-1, -2)  # D(g)† = D(g^{-1})
        coeffs[l] = np.einsum("abg,abgmn,abg->mn", samples, D_dag, weights, optimize=True)
    return coeffs


def inverse_fourier_on_grid(
    coeffs: dict[int, np.ndarray], D_grid: dict[int, np.ndarray], l_max: int
) -> np.ndarray:
    """
    Inverse Peter–Weyl transform on the Euler grid:

      f(g) = Σ_{l=0..L_MAX} (2l+1) Tr[ f̂^l D^l(g) ].
    """
    shape = next(iter(D_grid.values())).shape[:3]
    out = np.zeros(shape, dtype=np.complex128)
    for l in range(l_max + 1):
        D = D_grid[l]
        out += (2 * l + 1) * np.einsum("mn,abgnm->abg", coeffs[l], D.swapaxes(-1, -2), optimize=True)
    return out.real


# =========================================================
# Example signal + kernel (defined directly on rotation matrices)
# =========================================================

def rotation_angle(rotations: np.ndarray) -> np.ndarray:
    """Geodesic rotation angle in [0, π] from a rotation matrix."""
    tr = np.trace(rotations, axis1=-2, axis2=-1)
    cos_angle = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_angle)


def create_signal(rotations: np.ndarray) -> np.ndarray:
    """
    A smooth, asymmetric signal on SO(3).

    Defined in a coordinate-free way via the rotation matrix entries so it can be
    evaluated at any group element without Euler-angle singularities.
    """
    # Columns are the images of the basis vectors under the rotation.
    x_axis = rotations[..., :, 0]
    y_axis = rotations[..., :, 1]
    z_axis = rotations[..., :, 2]

    return (
        0.8 * z_axis[..., 0]
        + 0.4 * x_axis[..., 1]
        - 0.3 * y_axis[..., 0]
        + 0.2 * (z_axis[..., 2] ** 3)
    )


def create_kernel(rotations: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    """
    A localized (but not conjugation-invariant) kernel around the identity.

    The isotropic part depends on the rotation angle, and we add a small
    anisotropic factor so that equivariance requires the correct multiplication
    order in Fourier space.
    """
    ang = rotation_angle(rotations)
    isotropic = np.exp(-(ang**2) / (2 * sigma**2))
    anisotropic = 1.0 + 0.4 * rotations[..., 0, 0]  # stays positive in [0.6, 1.4]
    return isotropic * anisotropic


def normalize_kernel(kernel: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normalize so ∫ L(h) dμ(h) ≈ 1 under the discrete quadrature."""
    return kernel / (np.sum(kernel * weights) + 1e-12)


def spectral_convolution(
    f_coeffs: dict[int, np.ndarray], L_coeffs: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """(f ⋆ L)̂^l = L̂^l f̂^l for the convention used in this file."""
    return {l: (L_coeffs[l] @ f_coeffs[l]) for l in f_coeffs}


# =========================================================
# Main
# =========================================================

def main() -> None:
    alpha, beta, gamma, weights, rotations = make_euler_zyz_grid(N_ALPHA, N_BETA, N_GAMMA)

    f = create_signal(rotations)
    L = normalize_kernel(create_kernel(rotations, sigma=0.6), weights)

    D_grid = precompute_wigner_D_grid(alpha, beta, gamma, L_MAX)

    print("Computing spectral convolution on SO(3)...")
    f_hats = fourier_coefficients_from_samples(f, D_grid, weights)
    L_hats = fourier_coefficients_from_samples(L, D_grid, weights)

    conv_hats = spectral_convolution(f_hats, L_hats)
    conv = inverse_fourier_on_grid(conv_hats, D_grid, L_MAX)
    print("Done.")

    # -----------------------------------------------------
    # Equivariance check (left translations)
    # -----------------------------------------------------
    print("\n--- Equivariance Check (left translation) ---")
    R_shift = Rotation.from_euler("ZYZ", SHIFT_EULER_ZYZ).as_matrix()

    # f_shifted(g) = f(R_shift^{-1} g) sampled on the same Euler grid.
    rotations_flat = rotations.reshape(-1, 3, 3)
    rotations_shifted = np.einsum("ij,pjk->pik", R_shift.T, rotations_flat, optimize=True)
    rotations_shifted = rotations_shifted.reshape(rotations.shape)
    f_shifted = create_signal(rotations_shifted)

    # Convolve shifted input
    f_shifted_hats = fourier_coefficients_from_samples(f_shifted, D_grid, weights)
    conv_shifted_input_hats = spectral_convolution(f_shifted_hats, L_hats)
    conv_shifted_input = inverse_fourier_on_grid(conv_shifted_input_hats, D_grid, L_MAX)

    # Shift the convolved output using the *representation* action on coefficients.
    D_shift_inv = {
        l: np.conj(wigner_D_matrix(l, *SHIFT_EULER_ZYZ)).T for l in range(L_MAX + 1)
    }
    conv_shifted_output_hats = {l: (conv_hats[l] @ D_shift_inv[l]) for l in range(L_MAX + 1)}
    conv_shifted_output = inverse_fourier_on_grid(conv_shifted_output_hats, D_grid, L_MAX)

    equiv_err = float(np.max(np.abs(conv_shifted_input - conv_shifted_output)))
    scale = float(np.max(np.abs(conv)) + 1e-12)
    print(f"Max equivariance error = {equiv_err:.3e}")
    print(f"Relative error (scaled by max |output|) = {(equiv_err / scale):.3e}")

    # Also show how well the discrete quadrature captures the Fourier translation rule.
    hat_errs = []
    for l in range(L_MAX + 1):
        predicted = f_hats[l] @ D_shift_inv[l]
        hat_errs.append(np.max(np.abs(f_shifted_hats[l] - predicted)))
    print(f"Max Fourier translation error over ℓ≤{L_MAX} = {max(hat_errs):.3e}")


if __name__ == "__main__":
    main()
