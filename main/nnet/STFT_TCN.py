import numpy as np
from scipy.signal import spectrogram
# import random

# Generate a sample signal
fs = 16000  # Sampling frequency (Hz)
# t = np.arange(0, 5, 1/fs)  # Time vector from 0 to 5 seconds
# x = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*50*t)  # A signal with two frequency components
x = np.random.rand(2, 1000)
# Define STFT parameters
nperseg = 64  # Frame size
noverlap = 32  # Frame shift
nfft = 512  # FFT size

# Compute the STFT
frequencies, times, stft_matrix = spectrogram(x, fs, window='hamming', nperseg=nperseg, noverlap=noverlap, nfft=nfft)

# Extract magnitude information from the STFT
magnitude = np.abs(stft_matrix)
print("frequencies", frequencies)
print(frequencies.shape)
print("times", times)
print(times.shape)
print("magnitude", magnitude)
print(magnitude.shape)

# Define the window size for dimensionality reduction
window_size = 16

# Calculate the number of windows
num_windows = magnitude.shape[1] // window_size

print("num_windows", num_windows)

# Initialize an empty feature vector
feature_vector = np.zeros((magnitude.shape[0], num_windows))

# Apply dimensionality reduction (e.g., averaging) for each window
for i in range(num_windows):
    start_idx = i * window_size
    end_idx = start_idx + window_size
    windowed_magnitude = magnitude[:, start_idx:end_idx]
    feature_vector[:, i] = np.mean(windowed_magnitude, axis=None)

# Ensure the feature vector has a dimensionality of 256
if feature_vector.shape[1] > 256:
    feature_vector = feature_vector[:, :256]  # Truncate to 256 dimensions
elif feature_vector.shape[1] < 256:
    # You may need to pad or interpolate if you want exactly 256 dimensions
    feature_vector = np.pad(feature_vector, ((0, 0), (0, 256 - feature_vector.shape[1])), 'constant')

# Now 'feature_vector' contains your 256-dimensional feature with a window size of 16
print("feature_vector:", feature_vector)
print(feature_vector.shape)


