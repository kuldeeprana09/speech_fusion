import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Load the clean and enhanced audio files (replace 'clean.wav' and 'enhanced.wav' with your file names)
# clean_audio, sr_clean = librosa.load('/media/speech70809/Data011/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_-3dB/mix/CX_20160429_001_aligned_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr-3_fileid_482.wav', sr=None)
# enhanced_audio, sr_enhanced = librosa.load('/media/speech70809/Data011/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_-3dB/s1/CX_20160429_001_aligned_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr-3_fileid_482.wav', sr=None)

noisy_speech, sr_noisy = librosa.load('/media/speech70809/Data011/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_-3dB/mix/CX_20160429_001_aligned_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr-3_fileid_482.wav', sr=None)
clean_speech, sr_clean  = librosa.load('/media/speech70809/Data011/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_-3dB/s1/CX_20160429_001_aligned_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr-3_fileid_482.wav', sr=None)
# BASE_speech, sr_base  = librosa.load('/media/speech70809/Data011/speech_donoiser_new/spectogram_scl_paper/bus_ms_sl1/CX_20160429_001_aligned_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr-3_fileid_482.wav', sr=None)
# BASE3_speech, sr_base3  = librosa.load('/media/speech70809/Data011/speech_donoiser_new/spectogram_scl_paper/bus_ms_sl3/CX_20160429_001_aligned_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr-3_fileid_482.wav', sr=None)
# CWTCN_speech, sr_cwtcn  = librosa.load('/media/speech70809/Data011/speech_donoiser_new/spectogram_scl_paper/bus_ch_cl1/CX_20160429_001_aligned_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr-3_fileid_482.wav', sr=None)
CWMTCN_speech, sr_cwmtcn  = librosa.load('/media/speech70809/Data011/speech_donoiser_new/spectogram_scl_paper/bus_ch_sl3/CX_20160429_001_aligned_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr-3_fileid_482.wav', sr=None)

# Compute the spectrograms using Short-Time Fourier Transform (STFT)
n_fft = 2048  # Number of FFT points (adjust to your needs)
hop_length = 512  # Hop size between consecutive frames (adjust to your needs)

noisy_speech = np.abs(librosa.stft(noisy_speech, n_fft=n_fft, hop_length=hop_length))
clean_speech = np.abs(librosa.stft(clean_speech, n_fft=n_fft, hop_length=hop_length))
# BASE_speech = np.abs(librosa.stft(BASE_speech, n_fft=n_fft, hop_length=hop_length))
# BASE3_speech = np.abs(librosa.stft(BASE3_speech, n_fft=n_fft, hop_length=hop_length))
# CWTCN_speech = np.abs(librosa.stft(CWTCN_speech, n_fft=n_fft, hop_length=hop_length))
CWMTCN_speech = np.abs(librosa.stft(CWMTCN_speech, n_fft=n_fft, hop_length=hop_length))

# Convert amplitude to dB scale for better visualization
noisy_speech = librosa.amplitude_to_db(noisy_speech, ref=np.max)
clean_speech = librosa.amplitude_to_db(clean_speech, ref=np.max)
# BASE_speech = librosa.amplitude_to_db(BASE_speech, ref=np.max)
# BASE3_speech = librosa.amplitude_to_db(BASE3_speech, ref=np.max)
# CWTCN_speech = librosa.amplitude_to_db(CWTCN_speech, ref=np.max)
CWMTCN_speech = librosa.amplitude_to_db(CWMTCN_speech, ref=np.max)
# Plot the spectrograms side by side
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# librosa.display.specshow(noisy_speech, sr=sr_clean, hop_length=hop_length, x_axis='time', y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Noisy Speech')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency')

# plt.subplot(1, 2, 2)
# librosa.display.specshow(clean_speech, sr=sr_clean, hop_length=hop_length, x_axis='time', y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Clean Speech')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency')
# plt.show()
# This code will plot the spectrograms of the clean and enhanced audio side by side. The x-axis represents time in seconds, the y-axis represents frequency, and the color scale represents the magnitude of the spectrogram values in decibels (dB). Brighter regions indicate higher energy in the signal.

# Visual inspection of the spectrograms can give you a quick understanding of the differences between the clean and enhanced speech signals, particularly regarding noise reduction and speech clarity.
# Create a 2x3 grid for subplots
# Create a 2x3 grid for subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Plot the spectrograms in each subplot
librosa.display.specshow(noisy_speech, sr=sr_clean, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[0, 0])
axes[0, 0].set_title('Noisy Speech')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Frequency')

librosa.display.specshow(clean_speech, sr=sr_clean, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[0, 1])
axes[0, 1].set_title('Clean Speech')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Frequency')
# librosa.display.specshow(BASE_speech, sr=sr_clean, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[0, 2])
# axes[0, 1].set_title('BASE')
# axes[0, 1].set_xlabel('Time (s)')
# axes[0, 1].set_ylabel('Frequency')
# librosa.display.specshow(BASE3_speech, sr=sr_clean, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[1, 0])
# axes[1, 0].set_title('BASE-3')
# axes[1, 0].set_xlabel('Time (s)')
# axes[1, 0].set_ylabel('Frequency')
# librosa.display.specshow(CWTCN_speech, sr=sr_clean, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[1, 1])
# axes[1, 1].set_title('CW_TCN')
# axes[1, 1].set_xlabel('Time (s)')
# axes[1, 1].set_ylabel('Frequency')
librosa.display.specshow(CWMTCN_speech, sr=sr_clean, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[0, 2])
axes[0, 2].set_title('CWM_TCN3')
axes[0, 2].set_xlabel('Time (s)')
axes[0, 2].set_ylabel('Frequency')
# Adjust layout and spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()
# In this example, we create a 2x3 grid of subplots using plt.subplots(2, 3). Then, we plot different data in each subplot by accessing each individual subplot using the axes array. The plt.tight_layout() function is used to adjust the spacing between subplots so that they do not overlap.

# You can replace the y1, y2, ..., y6 arrays with your data for each subplot. Customize the plots as needed to visualize the specific data you want to compare inside the single plt.figure.










