import numpy as np

from scipy.signal import find_peaks
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

def psola2(sample, peaks, scale):
    new_signal = np.zeros(int(len(sample)*scale)+10)
    overlap = 0.5

    for x in range(len(peaks)-1):
        period = peaks[x+1] - peaks[x]
        new_period = int(period * scale)
        z = int(peaks[x] * scale)

        hwindow = np.hamming(int(period + overlap*period*2))
        i = 0
        u = -int(period*overlap)
        for y in range(peaks[x]-int(period*overlap), peaks[x]):
            if z+u > 0 and z+u < len(new_signal):
                new_signal[z+u] += sample[y] * hwindow[i]
                i += 1
                u += 1

        u = 0
        for y in range(peaks[x], peaks[x]+period):
            if z+u > 0 and z+u < len(new_signal):
                new_signal[z+u] += sample[y] * hwindow[i]
                i += 1
                u += 1

        #overlap
        u = int(peaks[x]+period)
        for y in range(peaks[x]+period, peaks[x]+int(period*overlap)):
            if z+u > 0 and z+u < len(new_signal):
                new_signal[z+u] += sample[y] * hwindow[i]
                i += 1
                u += 1
    return new_signal


def get_periods(sample):
    freqs = fft(sample)

    freqs[20000:len(freqs)] = 0

    y = ifft(freqs).real

    peaks, properties = find_peaks(y)

    return peaks