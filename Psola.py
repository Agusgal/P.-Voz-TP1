import numpy as np
import matplotlib.pyplot as plt
from LPC.Autocorrealciones import autocorr

import librosa as lr

if __name__ == '__main__':

    audio, fs = lr.load('04 Harder, Better, Faster, Stronger.wav', sr=44100, offset=0.1, duration=20)

    print(audio)

    r = autocorr(audio)
    print(r)


    time = np.arange(0, len(audio)) / fs

    fig, ax = plt.subplots()
    ax.plot(time, audio)
    ax.set(xlabel='time (s)', ylabel='Sound amplitude')
    plt.show()
    pass


