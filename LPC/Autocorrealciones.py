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
    L *= 10e-3 * 44100

    # ventana rectangular
    window = [i >= n and i <= n + L - 1 for i in range(len(sgn))]

    # arreglo nuevo
    arr_nuevo = np.multiply(sgn, window)
    corr = signal.correlate(arr_nuevo, arr_nuevo, 'full')

    return corr


def LPC(sgn, M):
    r = corr_short_time(sgn, len(sgn) // 2, 20)
    center_r = len(r) // 2

    r_M = r[center_r: center_r + M]
    r_M2 = r[center_r + 1: center_r + M + 1]

    alphas = solve_toeplitz(r_M, r_M2)
    return alphas


def plot_psd(signals, fs, Ms):
    plt.figure(figsize=(12, 12))
    plt.grid(which='both')

    plt.title('Espectro de señal de error para diferentes filtros')

    for signal, M in zip(signals, Ms):
        signal_asd = np.fft.fft(signal, n=len(signal))

        signal_psd = np.abs(signal_asd) ** 2

        freqs = np.fft.fftfreq(len(signal), d=1 / fs)

        plt.semilogx(freqs, signal_psd, label=f'M = {M}')

    plt.legend()

if __name__ == '__main__':
    filename = 'DaftPunk.wav'
    filename2 = '../Resources/Lorde.wav'
    sample, sr = librosa.load(filename2, sr=44100)

    plt.plot(np.linspace(0, 30, len(sample)), sample)
    plt.show()

    alphas = LPC(sample, 20)

    b = np.hstack([[0], -1 * alphas[1:]])
    sample_hat = signal.lfilter(b, [1], sample)

    fig, ax = plt.subplots()
    ax.plot(sample)
    ax.plot(sample_hat, linestyle='--')
    ax.legend(['y', 'y_hat'])
    ax.set_title('LP Model Forward Prediction')
    plt.show()

    b2 = np.hstack([[1], alphas[1:]])
    err = signal.lfilter(b2, [1], sample)
    plt.plot(np.linspace(0, 30, len(sample)), err)
    plt.show()

    alphasLibrosa = librosa.lpc(sample, 5)
    b3 = np.hstack([[1], -1 * alphasLibrosa[1:]])
    sampleLibrosa = signal.lfilter(b3, [1], sample)


