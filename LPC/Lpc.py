import numpy as np
from LPC.Autocorrealciones import autocorr

def lpc(signal, order):
    r = autocorr(signal)

    return r




