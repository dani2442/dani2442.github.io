"""
SO(2) Group Convolution (Discrete sanity checks)
================================================

This script implements a simple group convolution on SO(2) (the circle) and
verifies two standard properties:

1) Direct sum vs FFT convolution agree (up to floating-point error).
2) Left-equivariance: shifting the input then convolving equals convolving then
   shifting the output.

If you're looking for an *animation* of the SO(2) equivariance property, see:
`group-cnn/visualization_so2.py`.
"""

import numpy as np

N = 32
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Discretize SO(2) ≅ [0, 2π)
dtheta = 2 * np.pi / N

f = np.sin(3 * theta) + 0.5 * np.cos(5 * theta)  # Example signal f(θ) on the circle
L = np.exp(-0.5 * (((theta + np.pi) % (2 * np.pi) - np.pi) / 0.5) ** 2)


# Method A: Direct discretized integral (O(N^2))
# (Ff)(θ_k) ≈ (1/(2π)) Σ_{j=0}^{N-1} f(θ_k - θ_j) L(θ_j) dθ

conv_direct = np.empty(N)
j = np.arange(N)

for k in range(N):
    idx = (k - j) % N  # f(θ_k - θ_j) corresponds to f[(k-j) mod N] on the grid
    conv_direct[k] = (1 / (2 * np.pi)) * np.sum(f[idx] * L) * dtheta

# Method B: Fourier / Peter–Weyl (FFT) (O(N log N))
# ifft(fft(f) * fft(L))[k] = Σ f[k-j] L[j]
# We still multiply by 1/(2π) to match the integral's factor.

f_hat = np.fft.fft(f)
L_hat = np.fft.fft(L)

conv_fft = np.fft.ifft(f_hat * L_hat).real * dtheta / (2 * np.pi)


# ---------------------------------------------------------
# 3) Check agreement
# ---------------------------------------------------------
max_err = np.max(np.abs(conv_direct - conv_fft))
print("Max |direct - fft| =", max_err)

# ---------------------------------------------------------
# 4) Optional: Equivariance check (commutes with rotations)
#
# Left-translation on SO(2): (λ(α)f)(θ) = f(θ - α)
# On the grid: shift by s steps => f_shift[k] = f[(k-s) mod N]
# which is np.roll(f, s).
# ---------------------------------------------------------
s = 17  # shift steps
f_shift = np.roll(f, s)

# Convolve shifted input (direct, same formula)
conv_shift_direct = np.empty(N)
for k in range(N):
    idx = (k - j) % N
    conv_shift_direct[k] = (1/N) * np.sum(f_shift[idx] * L)

# Compare to shifting the convolved output
equiv_err = np.max(np.abs(conv_shift_direct - np.roll(conv_direct, s)))
print("Equivariance error =", equiv_err)
