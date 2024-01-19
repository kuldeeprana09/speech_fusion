import torch
import torchaudio
import numpy as np

def stoi(clean_audio, enhanced_audio, sample_rate):
    # Convert audio signals to tensors
    clean_audio_tensor = torch.from_numpy(clean_audio).unsqueeze(0)
    enhanced_audio_tensor = torch.from_numpy(enhanced_audio).unsqueeze(0)

    # Resample audio signals if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        clean_audio_tensor = resampler(clean_audio_tensor)
        enhanced_audio_tensor = resampler(enhanced_audio_tensor)

    # Compute STOI score
    stoi_transform = torchaudio.transforms.Spectrogram(n_fft=256, hop_length=128, win_length=256, window_fn=torch.hann_window)
    clean_spec = stoi_transform(clean_audio_tensor)
    enhanced_spec = stoi_transform(enhanced_audio_tensor)
    numerator = torch.sum(clean_spec * enhanced_spec, dim=(1, 2))
    denominator = torch.sqrt(torch.sum(clean_spec ** 2, dim=(1, 2))) * torch.sqrt(torch.sum(enhanced_spec ** 2, dim=(1, 2)))
    eps = torch.finfo(torch.float32).eps
    score = torch.mean(numerator / (denominator + eps))

    # Convert score to numpy array and return
    return np.array(score.detach())