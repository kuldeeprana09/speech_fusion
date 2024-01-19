# import numpy as np

# def overlap_add(signal, window_size, overlap):
#   """Performs overlap-add windowing on a signal.

#   Args:
#     signal: A NumPy array containing the signal to be windowed.
#     window_size: The size of the window to use.
#     overlap: The amount of overlap between the windows.

#   Returns:
#     A NumPy array containing the windowed signal.
#   """

#   # Create a window function.
#   window = np.hanning(window_size)

#   # Divide the signal into windows.
#   windows = np.array([signal[i:i+window_size] for i in range(0, len(signal), window_size - overlap)])

#   # Apply the window function to each window.
#   windowed_windows = windows * window

#   # Overlap and add the windows.
#   output = np.zeros(len(signal))
#   for i in range(len(windows)):
#     output[i:i+window_size] += windowed_windows[i]

#   return output

# # Example usage:

# signal = np.random.randn(1024)
# window_size = 256
# overlap = 128

# windowed_signal = overlap_add(signal, window_size, overlap)

# # The windowed signal can now be processed using any desired function.


import numpy as np

def overlap_add(signal, window, hop_size):
    signal_length = len(signal)
    print("signal_length",signal_length)
    window_length = len(window)
    print("window_length", window_length)
    result = np.zeros(signal_length)
    print("result", result)

    for i in range(0, signal_length, hop_size):
        if i + window_length > signal_length:
            break
        segment = signal[i:i + window_length]
        print("segment", segment)
        result[i:i + window_length] += segment * window

    return result

# Example usage:
# Generate a sample signal and window
signal_length = 1000
signal = np.random.rand(signal_length)
window_length = 100
window = np.hanning(window_length)

# Set the hop size (overlap size)
hop_size = window_length // 2

# Apply overlap-add
result_signal = overlap_add(signal, window, hop_size)
# print(result_signal)