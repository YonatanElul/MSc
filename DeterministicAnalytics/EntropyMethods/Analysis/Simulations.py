from DeterministicAnalytics.EntropyMethods.Utilities.GeneratePinkNoise import generate_pink_noise
from DeterministicAnalytics.EntropyMethods.Utilities.NoiseGenerator import white
from DeterministicAnalytics.EntropyMethods.Utilities.CalcEnt import EntropyManipulator

import matplotlib.pyplot as plt
import numpy as np
import pickle


# Define the length of each noise signal
n_noise_samples = 10000

# Generate Pink Noise
pink_noise, pink_freqs, pink_spectrum = generate_pink_noise(n_noise_samples)
inds = np.arange(pink_noise.size)

# Generate White Noise
white_noise = white(num_points=n_noise_samples)

# Plot the noise signals
plt.figure()
plt.plot(inds, pink_noise)
plt.plot(inds, white_noise)

# Calculate the noise entropy measure
plt.figure()
entropy_calculator = EntropyManipulator()
p_time, pink_entropy = entropy_calculator.calc_multi_scale_entropy_moment(inds, pink_noise)
w_time, white_entropy = entropy_calculator.calc_multi_scale_entropy_moment(inds, white_noise)
plt.plot(p_time, pink_entropy)
plt.plot(w_time, white_entropy)

# Load Real Records Data
entropies_path = r"B:\Studies\Master's\CardiacDiagnostics\Database\EntropyDatabase.pkl"
data = pickle.load(open(entropies_path, 'rb'))
entropies = data['Entropies']

# TODO: Apply the same smoothing method from 'Multiscale Entropy Analysis of Complex Physiologic Time Series' to our
#       data

for l in entropies:
    plt.figure()

    for time, ent in l:
        # print("Current Entropy Mean: {}".format(np.mean(ent)))
        # print("Current Entropy STD: {}".format(np.std(ent)))
        # ent[ent > 50] = np.mean(ent)
        plt.plot(time, ent)

plt.show()





