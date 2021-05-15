import numpy as np
from scipy.fftpack import fft, ifft


def autocorr(signal):
    """

    :param signal:
    :return:
    """
    maxlag = signal.shape[-1]
    fftPoints = nextpow2(2*maxlag - 1)
    a = computeAcorr(signal, fftPoints, maxlag)
    return a


def computeAcorr(signal, fftpoints, maxlag):
    a = np.real(ifft(np.abs(fft(signal, n=fftpoints) ** 2))) ##Aca no se si dividir por el tamaño
    return a[..., :maxlag + 1]


def nextpow2(n):
    """
    Devuelve la potencia de 2 tal que 2^p >= n

    :param n:
    :return:
    """

    f, p = np.frexp(n)
    if f == 0.5:
        return p - 1
    elif np.isfinite(f):
        return p
    else:
        return f
