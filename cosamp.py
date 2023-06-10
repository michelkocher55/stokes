# From https://github.com/rfmiotto/CoSaMP/blob/master/src/cosamp/cosamp.py
# MIT licenced
# Adapted by THNO
import numpy as np

def cosamp(Phi, u, s, tol=1e-10, max_iter=1000):
    """
    @Brief:  "CoSaMP: Iterative signal recovery from incomplete and inaccurate
             samples" by Deanna Needell & Joel Tropp

    @Input:  Phi - Sampling matrix
             u   - Noisy sample vector
             s   - Sparsity vector

    @Return: A s-sparse approximation "a" of the target signal
    """
    max_iter -= 1 # Correct the while loop
    num_precision = 1e-12
    a = np.zeros(Phi.shape[1], dtype=np.complex_)
    v = u
    iter_ = 0
    halt = False
    while not halt:
        iter_ += 1
        # Originally, the code couldn't deal with complex matrices since a transpose
        # was performed instead of a conjugate transpose.
        #y = np.abs(np.dot(np.transpose(Phi), v))
        """
        NOTE: @thno: v is the vector approaching u in this procedure.
            This dot product may be performed as a FFT followed by a wavelet decomposition...
        """
        y = np.abs(np.dot(Phi.conj().T, v))
        Omega = [i for (i, val) in enumerate(y) if val > np.sort(y)[::-1][2*s] and val > num_precision] # equivalent to below
        #Omega = np.argwhere(y >= np.sort(y)[::-1][2*s] and y > num_precision)
        T = np.union1d(Omega, a.nonzero()[0])
        #T = np.union1d(Omega, T)

        """
        NOTE: @thno: pinv is a least-square solution to a subset of Phi and vector u.
            u here is constant and is our measurement vector (1D)
        """
        # Least squares and pinv are equivalent
        b = np.linalg.lstsq(Phi[:,T], u)[0]
        # Original code uses a pseudo-inverse
        #b = np.dot( np.linalg.pinv(Phi[:,T]), u )

        iGood = (abs(b) > np.sort(abs(b))[::-1][s]) & (abs(b) > num_precision)
        T = T[iGood]
        a[T] = b[iGood]
        v = u - np.dot(Phi[:,T], b[iGood])
        err = np.linalg.norm(v)/np.linalg.norm(u)
        print("Iteration {} done, err = {}\r".format(iter_, err))

        halt = err < tol or \
               iter_ > max_iter

    return a
