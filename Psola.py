import numpy as np
import matplotlib.pyplot as plt

import librosa as lr

if __name__ == '__main__':

    audio, fs = lr.load('04 Harder, Better, Faster, Stronger.wav', 44100)

    time = np.arange(0, len(audio)) / fs

    fig, ax = plt.subplots()
    ax.plot(time, audio)
    ax.set(xlabel='time (s)', ylabel='Sound amplitude')
    plt.show()



def findPitchPoints(signal, fs, )
