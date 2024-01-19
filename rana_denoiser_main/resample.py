import librosa
import soundfile as sf
import os
# os.chdir(r'speech_donoiser_new/datasets/Reverb_noise_new')
target_path = r'speech_donoiser_new/datasets/Reverb_noise'

for wav in os.listdir(target_path):
    print(wav)
    target = target_path + "/" + wav
    audio, sr = librosa.load(target)
    auudi_16k = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
    sf.write(wav, auudi_16k, 16000)
