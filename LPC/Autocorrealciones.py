from scipy.linalg import solve_toeplitz, toeplitz
import librosa


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def corr_short_time(sgn, n, L):
    """
    Parameters
    ----------
    sgn: np.array, señal a analizar
    n:   escalar, punto donde se calcula (n hat)
    L:   tamaño de bloque (ms)

    Return
    ----------
    retorna correlacion short time en instante n
    """
    pos = L * 10e-3 * 44100

    # ventana rectangular
    # window = [i >= n and i <= n + L - 1 for i in range(len(sgn))]

    # Hamming
    window = np.zeros(len(sgn))
    hamming = np.hamming(L)
    window[n:n + L] = hamming

    # arreglo nuevo
    arr_nuevo = np.multiply(sgn, window)
    corr = signal.correlate(arr_nuevo, arr_nuevo, 'full')

    return corr


def LPC(sgn, M):
    r = corr_short_time(sgn, len(sgn) // 2, 25)
    center_r = len(r) // 2

    r_M = r[center_r: center_r + M]
    r_M2 = r[center_r + 1: center_r + M + 1]

    alphas = solve_toeplitz(r_M, r_M2)
    return alphas


def overlap_add(sgn_arr):
    # sgn_arr: signal array
    pass


def plot_psd(signals, fs, Ms):
    plt.figure(figsize=(12, 12))
    plt.grid(which='both')

    plt.title('Espectro de señal de error para diferentes filtros')

    for signal, M in zip(signals, Ms):
        signal_asd = np.fft.fft(signal, n=len(signal))

        signal_psd = np.abs(signal_asd) ** 2

        freqs = np.fft.fftfreq(len(signal), d=1 / fs)

        plt.semilogx(freqs, signal_psd, label=f'M = {M}')


if __name__ == '__main__':
    filename = '../Resources/DaftPunk.wav'
    filename2 = '../Resources/Lorde.wav'
    sample, sr = librosa.load(filename, sr=44100)
    sample2, sr2 = librosa.load(filename2, sr=44100)



    alphas = LPC(sample2, 12)
    b = np.hstack((1, -alphas))

    alphasL = librosa.lpc(sample2, 20)
    bL = np.hstack((1, -alphasL[1:]))

    error = signal.lfilter(b, [1], sample2)
    sample_hat = signal.lfilter(alphas, [1], sample2)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(sample2, color='blue', label='Signal')
    ax.plot(error, color='black', label='Error')

    #plt.xlim([836700, 836850])
    plt.show()