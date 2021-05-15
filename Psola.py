import numpy as np
import matplotlib.pyplot as plt
from LPC.Autocorrealciones import autocorr

import librosa as lr

if __name__ == '__main__':

    arr = np.array([1, 2, 3])
    print(np.correlate(arr, arr, 'full'))




