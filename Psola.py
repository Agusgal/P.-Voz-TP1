import numpy as np
import matplotlib.pyplot as plt
from LPC.Autocorrealciones import autocorr

import librosa as lr

if __name__ == '__main__':

    arr = np.array([1, 2, 3])

    arr2 = np.array([[1, 2, 3], [4, 5, 6]])

    r = np.zeros(1, arr2.dtype)
    print(r)
    print(arr2.dtype)

    print(np.correlate(arr, arr, 'full'))




