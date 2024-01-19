import os
import torch
import torchaudio
from torchaudio.transforms import Convolution

# Set the directory paths for clean speech, reverb, and echo files
clean_speech_dir = 'speech_donoiser_new/datasets/NER_TRs_Vol_3_test'
ir_dir = 'speech_donoiser_new/datasets/reverb_noise_test'
echo_dir = '/media/speech70809/Data011/speech_donoiser_new/datasets/echowav'
output_dir = 'speech_donoiser_new/datasets/Reverb_noise_new'

# Get the list of clean speech file names
clean_speech_files = os.listdir(clean_speech_dir)

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set the parameters for the echo effect
delay_time = 0.5  # Adjust the delay time as desired
echo_strength = 0.6  # Adjust the echo strength as desired

# Iterate over each clean speech file
for clean_speech_file in clean_speech_files:
    # Load the clean speech waveform
    clean_speech_path = os.path.join(clean_speech_dir, clean_speech_file)
    clean_speech, sample_rate = torchaudio.load(clean_speech_path)

    # Find the corresponding reverb and echo file names
    reverb_file = clean_speech_file.replace('.wav', '_reverb.wav')
    echo_file = clean_speech_file.replace('.wav', '_echo.wav')

    # Load the reverb and echo waveforms
    reverb_path = os.path.join(reverb_dir, reverb_file)
    echo_path = os.path.join(echo_dir, echo_file)
    reverb, _ = torchaudio.load(reverb_path)
    echo, _ = torchaudio.load(echo_path)

    # Ensure the sample rates match
    if sample_rate != torchaudio.get_info(reverb_path).sample_rate or sample_rate != torchaudio.get_info(echo_path).sample_rate:
        raise ValueError("Sample rates of the audio files do not match.")

    # Normalize the clean speech, reverb, and echo waveforms
    clean_speech /= torch.max(torch.abs(clean_speech))
    reverb /= torch.max(torch.abs(reverb))
    echo /= torch.max(torch.abs(echo))

    # Create the Convolution transform for reverb effect
    convolution_reverb = Convolution(reverb.unsqueeze(0), stride=1)

    # Apply convolution for reverb effect
    reverb_speech = convolution_reverb(clean_speech.unsqueeze(0)).squeeze(0)

    # Set the delay length for echo effect (in samples)
    delay_samples = int(delay_time * sample_rate)

    # Apply the delay to the echo waveform
    delayed_echo = torch.zeros_like(clean_speech)
    delayed_echo[delay_samples:] = echo[:-(delay_samples)]

    # Adjust the echo strength
    echo_speech = echo_strength * delayed_echo

    # Add the echo to the reverb speech
    reverb_echo_speech = reverb_speech + echo_speech

    # Normalize the reverb echo speech
    reverb_echo_speech /= torch.max(torch.abs(reverb_echo_speech))

    # Save the reverb echo speech as a new audio file
    output_path = os.path.join(output_dir, clean_speech_file)
    
    torchaudio.save(output_path, reverb_echo_speech, sample_rate)
    