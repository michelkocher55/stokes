"""
Compressed sensing for ROFDR systems.
See reference papers:

Optical setup:
M.A. Farahani and T.~Gogolla, Spontaneous Raman scattering in optical fibers with modulated probe light for distributed temperature Raman remote sensing,
Journal of Lightwave Technology, 17(8):1379--1391, 1999-08.

CoSaMP recovery:
Deanna Needell and Joel A Tropp, CoSaMP: Iterative signal recovery from incomplete and inaccurate samples,
Applied and computational harmonic analysis, 26(3):301-321, 2009.

Compressed sensing basics:
Steven L Brunton and J Nathan Kutz, Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control, Cambridge University Press, 2022

Author: Thibault North, 2023
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import pywt as wt
from functools import partial

#import cvxpy as cvx # Unused for now
# For data import / export to Matlab
import scipy.io as sio

# CS reconstruction
from cosamp import cosamp


import os


"""
Random helpful notes
--------------------

np.fft.fft(wavelet_matrix(8, wavelet='haar'), norm='ortho', axis=0)
is equivalent to:
linalg.dft(8, scale='sqrtn') @ wavelet_matrix(8, wavelet='haar')

In [59]: np.sum(np.abs(np.fft.fft(wavelet_matrix(8, wavelet='haar'), norm='ortho', axis=0) - linalg.dft(8, scale='sqrtn') @ wavelet_matrix(8, wavelet='haar')))
Out[59]: 1.136332913096944e-14

What CoSaMP needs is PsiInv(N) which is just the DFT (normalized) of the wavelet_matrix(8, wavelet='haar').T.
This is also just the FFT of haarmatrix(N).T (where .T denotes the transpose operator)

See https://en.wikipedia.org/wiki/DFT_matrix
"""


# Some options
with_plots = 0
with_noise = 0
with_ramp = 0
use_cache = 0
save_matrices = 0

# Caching big matrices can be done...
CACHE_PREFIX = "cache"
from pathlib import Path
Path(CACHE_PREFIX).mkdir(exist_ok=True)

# Load and save to cache if required
# Just a decorator, can be ignored
def load_and_save(func):
    def wrapper(*args,**kwargs):
        wavelet = kwargs['wavelet'] if 'wavelet' in kwargs else None
        funcs = {"wavelet_matrix": "{}_{}.mat".format(N, wavelet),
                 "dft": "dft_{}".format(N),
                 "idft": "idft_{}".format(N),
                 "Psi": "psi_{}".format(N),
                 "PsiInv": "psiinv_{}".format(N)
                 }
        if not func.__name__ in funcs:
            print("Unknown function {}, recalculating".format(func.__name__))
        else:
            fn = funcs[func.__name__]
            cached_file = os.path.join(CACHE_PREFIX, fn)
            if use_cache and os.path.isfile(cached_file):
                cache_item = sio.loadmat(cached_file)
                print("Using cached file {}.mat".format(cached_file))
                return cache_item[fn]
            res = func(*args, **kwargs)
            if save_matrices:
                print("Saving matrix {}.mat".format(cached_file))
                sio.savemat(cached_file, {fn: res})
            return res

    return wrapper


# Generate a wavelet decomposition matrix for a given wavelet.
# Decomposition is done for log2(N) levels.
@load_and_save
def wavelet_matrix(N, wavelet='db2'):
    """
    Apply a wavelet decomposition on each basis vector of the identify matrix.
    Force the level to be log2(N)
    Force periodization to have as many coefficients as items in the signal
    """
    matrix = np.zeros((N, N))
    for i,k in enumerate(np.eye(N)):
        matrix[:,i] = wt.coeffs_to_array(wt.wavedec(k, wavelet, mode='periodization', level=int(np.log2(N))))[0]
    return matrix

# Generate a DFT matrix
@load_and_save
def dft(N):
    #return linalg.dft(N).T
    # Note: symmetric and orthogonal matrix, transpose is itself
    DFT = linalg.dft(N, scale='sqrtn')
    return DFT

# Generate an IDFT matrix
@load_and_save
def idft(N):
    #return linalg.dft(N).T
    # Note: symmetric and orthogonal matrix, transpose is itself
    IDFT = linalg.dft(N, scale='sqrtn').conj().T # scaling so that the matrix is orthogonal, ie tr(Q)Q = I
    return IDFT

# Compute the Psi matrix, which is the product of the wavelet transform matrix and
# the inverse Fourier transform
# In fact this function is never used, since CoSaMP needs its inverse
@load_and_save
def Psi(N):
    psi = wavelet_transform(N) @ idft(N)
    return psi


# Compute the Inverse of the Psi matrix, which is the product of the wavelet transform matrix and
# the inverse Fourier transform
# Recall C = AB ; inv(C) = inv(B) inv(A) => Psiinv = DFT(N) * H-1(N)
# This is all you need for compressed sensing!
@load_and_save
def PsiInv(N):
    return np.fft.fft(wavelet_transform(N).T, norm='ortho', axis=0)
    # The following would also work:
    # return linalg.dft(N, scale="sqrtn") @ wavelet_transform(N).T
    # return dft(N) @ wavelet_transform(N).T

# Program entry
if __name__ == "__main__":

    N = 1024 # Signal length
    p = int(N / 4)# N // 2 # subset of frequencies to use for reconstruction
    k_sparse = 50 # Sparsity (non-zero wavelet coefficients)

    wavelet = 'rbio1.1' # Works well
    wavelet = 'sym3' # Somehow works
    wavelet = 'haar'
    wavelet_transform = partial(wavelet_matrix, wavelet=wavelet)

    # Define time domain signal with some profiles, for Stokes and Anti-Stokes
    t = np.linspace(0, 1, N)
    stokes = np.exp(-(t + 1))
    astokes = np.exp(-(t + 1))
    # Starts and end at 0
    stokes[0] = 0
    astokes[0] = 0
    stokes[-N//100:] = 0
    astokes[-N//100:] = 0
    # add segment
    stokes[int(N / 10):int(N / 0.99)] += 0.01
    astokes[int(N / 10):int(N / 0.99)] += 0.01

    # add hotspot
    hotspots = {(15,20): 2, (40, 42): -1, (100,160):1, (200, 205):2}
    hotspots = {(int(N / 20), int(N / 19)): 2, (int(N / 8), int(N / 7)): -1, (int(N / 1.5), int(N / 1.4)):1, (int(N / 1.1), int(N / 1.05)):2}

    # hotspots = {(15,20): 2}
    fact = 200
    stokes_astokes_ratio = 5
    for (hotspot_start, hotspot_end), hotspot_ampl in hotspots.items():
        stokes[hotspot_start:hotspot_end] += hotspot_ampl / fact
        astokes[hotspot_start:hotspot_end] += hotspot_ampl / fact * stokes_astokes_ratio
        # Add a ramp on the signal
        if with_ramp:
            stokes[hotspot_start:hotspot_end] *= np.linspace(1, 1.1, hotspot_end - hotspot_start)
            astokes[hotspot_start:hotspot_end] *= np.linspace(1, 1.1, hotspot_end - hotspot_start)

    # add noise?
    if with_noise:
        stokes += np.random.randn(len(stokes)) / 3000
        astokes += np.random.randn(len(astokes)) / 3000

    print("Preparing stokes signals...")
    stokes_f = np.fft.fft(stokes)
    astokes_f = np.fft.fft(astokes)
    stokes_f = dft(N) @ stokes
    astokes_f = dft(N)  @ astokes

    # Substract Stokes / antiStokes
    stokes_diff = astokes_f - stokes_f

    # Remove the DC component
    dc = stokes_diff[0]
    stokes_diff[0] = 0
    stokes_f[0] = 0
    astokes_f[0] = 0
    # Retrieve stokes signal difference in the time domain for tests
    stokes_diff_t = np.fft.ifft(stokes_diff, norm='ortho')

    print("Generating H, IDF, Psi...")
    H = wavelet_transform(N)
    IDFT = idft(N)
    psi = Psi(N)

    # Already determine which items to keep for CS
    # Since the Fourier spectrum has first the DC component,
    # then the positive frequencies, then the negative ones,
    # we have to pick frequencies in the positive range
    # and their counterpart in the negative range
    # Never pick the DC component
    # Also don't pick the last one since we don't have its negative equivalent when N is
    # a power of two
    perm_positive = 1 + np.floor(np.random.rand(p // 2) * (int(( N - 1) / 2) - 2)).astype(int)

    # Experiment: use logspace sampling
    # This fixes the number of samples picked - Use a different base to change the sampling
    # Comment this out for a random sampling
    perm_positive = np.logspace(0, np.log10(N / 2), base=10, num=p//2).astype(int) + 1

    # Negative freqs must be the same as the positive ones chosen.
    perm_negative = [N - k for k in perm_positive]
    perm = np.concatenate([perm_positive, perm_negative])
    perm = np.unique(perm)

    wt_transform = wavelet_transform(N) @ stokes_diff_t # (just a check)
    wt_transform_pywt = np.concatenate(wt.wavedec(stokes_diff_t, wavelet, mode='periodization')).ravel() # (just a check)

    # Count number of nonzero items
    epsilon = 1e-6
    nonzero_hwt = len(np.where(np.abs(wt_transform.real) > epsilon)[0])

    # Check if the iDFT matrix brings us back in time (just a check)
    dt_fm = stokes_diff @ idft(N)

    # Construct the haar decomposition based on the frequency signal (just a check)
    hdec = psi @ stokes_diff

    # Now keep a subset of measurement and use compressed sensing
    y = stokes_diff[perm]

    # These two theta definitions are equivalent
    #theta =  psi.conj().T[perm, :]
    theta = PsiInv(N)[perm,:]

    # Save all for Matlab:
    # Uncomment this and add what would be useful
    """
    import scipy.io as sio
    ml_data =  {"theta": theta, "y": y}
    sio.savemat("theta_y.mat", ml_data)
    """

    # Run compressed sensing recovery CoSaMP, see cosamp.py file
    s = cosamp(theta, y, k_sparse, tol=1.e-10,max_iter=10)

    # Plot theoretical convergence
    # See Leonid P Yaroslavsky, How can one sample images with sampling rates close to the theoretical minimum?
    # Journal of Optics, 19(5):055706, 2017.
    Ks = np.arange(0, N / 2)[:, np.newaxis]
    Ms = np.arange(1, N + 1)
    conv_condition = Ms/Ks > -2* np.log(Ms / N)

    print("Generating figure...")
    plt.rcParams['figure.figsize'] = [12, 12]

    fig, ax = plt.subplots(nrows=4, ncols=2)
    ax[0,0].title.set_text("Stokes and anti-Stokes signal, time, {} samples".format(N))
    ax[0,0].plot(stokes, color='b')
    ax[0,0].title.set_text("Anti-Stokes signal, time")
    ax[0,0].plot(astokes, color='r')
    ax[1,0].title.set_text("Stokes signals, frequency, DC removed")
    ax[1,0].plot(np.fft.fftshift(10 * np.log10(np.abs(stokes_f)**2)), color='b', label='Stokes')
    ax[1,0].plot(np.fft.fftshift(10 * np.log10(np.abs(astokes_f)**2)), color='r', label='Anti-Stokes')
    ax[1,0].set_ylabel("Power spectral density")
    ax2y2 = ax[1,0].twinx()
    ax2y2.plot(np.fft.fftshift(np.unwrap(np.angle(stokes_f))), color='cornflowerblue', alpha=0.5, label='Stokes, phase')
    ax2y2.plot(np.fft.fftshift(np.unwrap(np.angle(astokes_f))), color='indianred', alpha=0.5, label='Anti-Stokes, phase')
    ax[1,0].legend()
    ax2y2.legend()
    ax[2,0].title.set_text("Stokes signals difference, frequency (no fftshift)")
    ax[2,0].plot(stokes_diff, color='g', label=' (DC) + {} positive + {} negative freqs'.format(N//2, N//2 -1))
    ax[2,0].plot(perm, stokes_diff[perm], 'x', label='Subset for CS: 2 * {} = {} freqs'.format(len(perm) // 2, len(perm)))
    ax[2,0].legend()

    ax[3,0].plot(stokes_diff_t, color='g', label='iFFT computed')
    ax[3,0].plot(IDFT @ stokes_diff, color='g', linestyle='dotted', label='IDFT matrix computed')
    ax[3,0].legend()
    ax[3,0].title.set_text("Stokes difference, time")

    ax[0,1].title.set_text("{} wavelet transform ({} / {} nonzero freqs, {:.1f}% sparsity)".format(wavelet, nonzero_hwt, N, 100 - 100* nonzero_hwt / N))
    ax[0,1].plot(wt_transform, color='m', label='matrix transform')
    ax[0,1].plot(wt_transform_pywt, color='m', linestyle='dotted', label='pywt transform')
    ax[0,1].legend()

    ax[1,1].title.set_text("Reconstruction from K={} samples ({} distinct freq out of {} => {:.1f}% of the freqs )".format(len(perm), len(perm) // 2, N // 2, 100 * len(perm) // 2 / (N // 2)))
    ax[1,1].plot(hdec, color='g', label='Target (full data)')
    ax[1,1].plot(s, color='darkorange', linewidth=2,label='Reconstructed')
    ax2y2 = ax[1,1].twinx()
    ax2y2.plot(100*(hdec - s) / hdec, alpha=0.5, color='y', label='Relative error')
    ax2y2.set_ylabel("Relative error [%]")
    ax[1,1].legend()

    ax[2,1].title.set_text("Inverse {} wavelet transform".format(wavelet))
    ax[2,1].plot(H.T @ hdec, color='g', label='Target (full data)')
    ax[2,1].plot(H.T @ s, color='darkorange', linewidth=2, label='Reconstructed')
    ax2y2 = ax[2,1].twinx()
    ax2y2.plot(100*(H.T @ hdec - H.T @ s) / (H.T @ hdec), alpha=0.5, color='y', label='Relative error')
    ax2y2.set_ylabel("Relative error [%]")
    ax[2,1].legend()

    ax[3,1].set_title("Convergence criterion")
    ax[3,1].imshow(conv_condition, cmap='gray', aspect='auto')
    ax[3,1].scatter(p, nonzero_hwt, marker="x", color="red", s=100, label='Attemped reconstruction')
    ax[3,1].scatter(len(perm), nonzero_hwt, marker="+", color="green", s=100, label='This reconstruction (unique measurements)')
    ax[3,1].set_xlabel("Number of measurement in CS")
    ax[3,1].set_ylabel("Sparsity")
    ax[3,1].legend()
    fig.tight_layout()
    plt.show()

# This function found somewhere online also generates Haar matrices...
# Unused now.
def haarmatrix(N):
    cached_file = os.path.join(CACHE_PREFIX, "haar_{}.mat".format(N))
    if use_cache and os.path.isfile(cached_file):
         H = sio.loadmat(cached_file)
         return H['Haar']

    n = np.floor(np.log(N)/np.log(2))

    if 2**n != N: raise Exception('error: size '+str(N)+' is not multiple of power of 2')

    z = np.resize(1.*np.arange(N)/N, (len(1.*np.arange(N)), len(1.*np.arange(N)/N)))
    k = np.transpose(np.resize(1.*np.arange(N), (len(1.*np.arange(N)/N), len(1.*np.arange(N)))))


    p  = np.floor(np.log(np.maximum(1,k))/np.log(2))
    q  = k - (2**p) + 1
    z1 = (q-1)   / (2**p)
    z2 = (q-0.5) / (2**p)
    z3 = q       / (2**p)
    A  = (1/np.sqrt(N)) * ((( 2**(p/2.)) * ((z >= z1) & (z < z2))) + ((-2**(p/2.)) * ((z >= z2) & (z < z3))))
    A[0,:] = 1/np.sqrt(N)

    # Save as temp
    sio.savemat(cached_file, {"Haar": A})
    return A
