from pesq import pesq
import os, glob
import soundfile
from tqdm import tqdm

refs = [os.path.basename(f) for f in sorted(glob.glob("/media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS/BUStt_03/s1/*.wav"))]
evals = [os.path.basename(f) for f in sorted(glob.glob("/media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/MS_SL2_R1_03dB_06dB_12dB/QUTCAR/QUTCARttnew_03/QUTCARttnew_03_tt/spk1/*.wav"))]

results = dict()
for i, (ref, eval) in tqdm(enumerate(zip(refs, evals))):
    assert ref == eval
    ref_path = os.path.join("/media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS/BUStt_03/s1", ref)
    eval_path = os.path.join("/media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/MS_SL2_R1_03dB_06dB_12dB/QUTCAR/QUTCARttnew_03/QUTCARttnew_03_tt/spk1", eval)
    y, sr = soundfile.read(ref_path)
    y_hat, sr = soundfile.read(eval_path)

    results[os.path.basename(ref)] = pesq(y, y_hat, sr)

# print(results)
print(sum(results.values())/len(results))


# from scipy.io import wavfile
# from pesq import pesq

# rate, ref = wavfile.read("/media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS/BUStt_03/s1")
# rate, deg = wavfile.read("./media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/MS_SL2_R1_03dB_06dB_12dB/QUTCAR/QUTCARttnew_03/QUTCARttnew_03_tt/spk1/")

# print(pesq(rate, ref, deg, 'wb'))
# print(pesq(rate, ref, deg, 'nb'))