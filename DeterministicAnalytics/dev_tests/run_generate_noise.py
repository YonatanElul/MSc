from DeterministicAnalytics.EntropyMethods.Utilities.GeneratePinkNoise import generate_pink_noise
from DeterministicAnalytics.EntropyMethods.Utilities.NoiseGenerator import pink, scipy_psd
import matplotlib.pyplot as plt
import numpy as np


# signal, frequencies, spectrum = generate_pink_noise(1000)
# inds = np.arange(signal.size)
#
# plt.figure()
# plt.plot(inds, signal)
#
# plt.figure()
# plt.plot(frequencies, np.abs(spectrum))

p_n = np.array(pink(1000))
f_axis, psd_of_x = scipy_psd(p_n, f_sample=1.0, nr_segments=4)

plt.figure()
plt.plot(f_axis, psd_of_x)

plt.show()
