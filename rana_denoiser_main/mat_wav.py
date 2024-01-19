import os
import scipy.io.wavfile as wavfile
import scipy.io

# Load the .mat file
mat_data = scipy.io.loadmat('/media/speech70809/Data011/speech_donoiser_new/datasets/mat_files/pink.mat')

# Extract the audio data from the .mat file
audio_data = mat_data['pink']

# Ensure the audio data is in the correct shape
if audio_data.ndim > 1:
    audio_data = audio_data.squeeze()

# Set the sample rate (modify if necessary)
sample_rate = 16000


output_dir = '/media/speech70809/Data011/speech_donoiser_new/datasets/noisex'
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
# Save the audio data as a .wav file
output_file = 'pink.wav'
noise_speech_path = os.path.join(output_dir, output_file)
wavfile.write(noise_speech_path, sample_rate, audio_data)