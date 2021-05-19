# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import librosa
import librosa.display

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, ifft
import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz, toeplitz
from scipy.signal import butter, lfilter
from scipy.signal import freqs


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a


def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_periods(sample):
    freqs = fft(sample)

    freqs[20000:len(freqs)] = 0

    y = ifft(freqs).real

    peaks, properties = find_peaks(y)

    return peaks


def psola(sample, peaks, scale):
    new_signal = np.zeros(len(sample))
    print(new_signal)
    for x in range(len(peaks)-1):
        period = peaks[x+1] - peaks[x]
        new_period = int(period * scale)

        if new_period <= period:
            new_signal[peaks[x]:period].sum(sample[peaks[x]:period])
        else:
            new_signal[peaks[x]:peaks[x+1]].sum(sample[peaks[x]:peaks[x+1]])
            new_signal[peaks[x+1]:period].sum([0 for x in range(new_period - period)])

    return new_signal


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


def compute_periods_per_sequence(signal, sequence, min_period, max_period):
    """
    Computes periodicity of a time-domain signal using autocorrelation
    :param sequence: analysis window length in samples. Computes one periodicity value per window
    :param min_period: smallest allowed periodicity
    :param max_period: largest allowed periodicity
    :return: list of measured periods in windows across the signal
    """
    N = len(signal)

    offset = 0  # current sample offset
    periods = []  # period length of each analysis sequence

    peak_counter = 0
    periods.append(peak_counter)

    while offset < N:
        fourier = fft(signal[offset: offset + sequence])
        fourier[0] = 0  # remove DC component
        autoc = ifft(fourier * np.conj(fourier)).real
        autoc_peak = min_period + np.argmax(autoc[min_period: max_period])

        peak_counter += autoc_peak

        periods.append(peak_counter)
        offset += sequence

    return periods


def LPC(sgn, M):
  r = corr_short_time(sgn, len(sgn)//2, 25)
  center_r = len(r)//2

  r_M = r[center_r: center_r + M]
  r_M2 = r[center_r + 1: center_r + M + 1]

  alphas = solve_toeplitz(r_M, r_M2)
  return alphas


filename = "05. Liability.wav"

sample, sr = librosa.load(filename, sr=44100)
sample2, sr2 = librosa.load(filename, sr=44100)


alphas = LPC(sample2, 12)
b = np.hstack((1, -alphas))

error = signal.lfilter(b, [1], sample2)

peaks = get_periods(sample)
#print(peaks)

new_signal = psola(sample, peaks, 1.1)

#fig, axs = plt.subplots(2, 1, sharex=True)

##axs[0].plot(sample)
#axs[0].plot(peaks, sample[peaks], 'X')
#axs[1].plot(new_signal, color="orange")
new_signal = butter_lowpass_filter(new_signal, 500, 44100, order=4)

plt.plot(new_signal)

plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
    #print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

