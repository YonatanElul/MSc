from numpy import concatenate, std, abs
from numpy.fft import ifft, fftfreq
from numpy.random import normal


def generate_pink_noise(samples: int = 100, exponent: float = 1.0, fmin: bool = False):
    """

    :param exponent:
    :param samples:
    :param fmin:
    :return:
    """

    # frequencies (we assume a sample rate of one)
    frequencies = fftfreq(samples)

    # scaling factor for all frequencies, though the fft for real signals is symmetric, the array with the results
    #  is not - take neg. half!
    s_scale = abs(concatenate([frequencies[frequencies < 0], [frequencies[-1]]]))

    # low frequency cutoff?!?
    if fmin:
        ix = sum(s_scale > fmin)

        if ix < len(frequencies):
            s_scale[ix:] = s_scale[ix]

    s_scale = s_scale ** (-exponent / 2.)

    # scale random power + phase
    sr = s_scale * normal(size=len(s_scale))
    si = s_scale * normal(size=len(s_scale))

    if not (samples % 2):
        si[0] = si[0].real

    spectrum = sr + 1J * si
    # this is complicated, because for odd sample numbers, there is one less positive freq than for even sample numbers
    spectrum = concatenate([spectrum[1 - (samples % 2):][::-1], spectrum[:-1].conj()])

    # time series
    signal = ifft(spectrum).real
    signal /= std(signal)

    return signal, frequencies, spectrum
