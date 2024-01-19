from pydub import AudioSegment
sound = AudioSegment.from_wav("/media/speech70809/Data011/speech_donoiser_new/datasets/Reverb_noise/WIDE-HALL-1.wav")
sound = sound.set_channels(1)
sound.export("/media/speech70809/Data011/speech_donoiser_new/datasets/Reverb_noise_new/WIDE-HALL-1", format="wav")
