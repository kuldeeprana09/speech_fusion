#import the pyplot and wavfile modules 

import matplotlib.pyplot as plot

from scipy.io import wavfile

 

# Read the wav file (mono)

# samplingFrequency, signalData = wavfile.read('y.wav')

samplingFrequency, mix_speech = wavfile.read('/media/speech70809/Data011/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_-3dB/mix/CX_20160429_002_aligned_BGD_150204_010_BUS.CH2_snr-3_fileid_1596.wav')
# print(samplingFrequency)
# print(mix_speech)
# print(len(mix_speech))
samplingFrequency, clean_speech = wavfile.read('/media/speech70809/Data011/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_-3dB/s1/CX_20160429_002_aligned_BGD_150204_010_BUS.CH2_snr-3_fileid_1596.wav')
# samplingFrequency, SL1_enhanced_speech = wavfile.read('')
# samplingFrequency, SL2_enhanced_speech = wavfile.read('/media/speech70809/Data011/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_-3dB/s1/CX_20160429_001_aligned_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr-3_fileid_482.wav')
# samplingFrequency, SL3_enhanced_speech = wavfile.read('/media/speech70809/TOSHIBA_3TB1/MS_R1_SL3_saved_models/BUS/BUStt_03/BUStt_03_tt/spk2/BW_20171124_020_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr3_fileid_154.wav')
# samplingFrequency, CH_SL1_enhanced_speech = wavfile.read('/media/speech70809/TOSHIBA_3TB1/MS_SL1_split_256channelWise_w4_16batch_models/BUS/BUStt_03/BUStt_03_tt/spk1/BW_20171124_020_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr3_fileid_154.wav')
# samplingFrequency, CH_SL2_enhanced_speech = wavfile.read('/media/speech70809/新增磁碟區/dB_model_backup_data_MS_SL1_SL2/MS_SL2_split_256channelWise_w4_16batch_models/BUS/BUStt_03/BUStt_03_tt/spk1/BW_20171124_020_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr3_fileid_154.wav')
samplingFrequency, CH_SL3_enhanced_speech = wavfile.read('/media/speech70809/Data011/speech_donoiser_new/spectogram_scl_paper/bus_ch_sl3/CX_20160429_002_aligned_BGD_150204_010_BUS.CH2_snr-3_fileid_1596.wav')

# ********************************Mahzoon Code******************************************************

#samplingFrequency, mix_speech = wavfile.read('/media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_-3dB/mix/CX_20160429_006_aligned_BGD_150204_020_BUS.CH2_snr-3_fileid_1704.wav')
# print(samplingFrequency)
# print(mix_speech)
# print(len(mix_speech))
#samplingFrequency, clean_speech = wavfile.read('/media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_-3dB/s1/CX_20160429_006_aligned_BGD_150204_020_BUS.CH2_snr-3_fileid_1704.wav')
# samplingFrequency, TCN1_FT_enhanced_speech = wavfile.read('/media/speech70809/TOSHIBA_3TB1/TCN1-FT/SpectogramAudiofiles/TCN1-FT/BW_20171124_020_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr3_fileid_154.wav')
# samplingFrequency, TCN2_FT_enhanced_speech = wavfile.read('/media/speech70809/TOSHIBA_3TB1/TCN1-FT/SpectogramAudiofiles/TCN2-FT/BW_20171124_020_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr3_fileid_154.wav')
# # samplingFrequency, SL3_enhanced_speech = wavfile.read('/media/speech70809/TOSHIBA_3TB1/MS_R1_SL3_saved_models/BUS/BUStt_03/BUStt_03_tt/spk2/BW_20171124_020_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr3_fileid_154.wav')
# samplingFrequency, CH_SL1_enhanced_speech = wavfile.read('/media/speech70809/TOSHIBA_3TB1/MS_SL1_split_256channelWise_w4_16batch_models/BUS/BUStt_03/BUStt_03_tt/spk1/BW_20171124_020_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr3_fileid_154.wav')
# samplingFrequency, CH_SL2_enhanced_speech = wavfile.read('/media/speech70809/新增磁碟區/dB_model_backup_data_MS_SL1_SL2/MS_SL2_split_256channelWise_w4_16batch_models/BUS/BUStt_03/BUStt_03_tt/spk1/BW_20171124_020_zaIfiS5-WiA-BGD_150204_040_BUS.CH2_snr3_fileid_154.wav')
# ********************************Mahzoon Code end******************************************************

plot.subplot(211)
plot.specgram(mix_speech,Fs=samplingFrequency)
plot.title("(a) Noisy Speech")
plot.xlim(0,5)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.savefig("Noisy_Speech.pdf", format="pdf", bbox_inches="tight")
plot.show()

plot.subplot(211)
plot.specgram(clean_speech,Fs=samplingFrequency)
plot.title("(b) Clean Speech ")
plot.xlim(0,5)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.savefig("Clean_Speech.pdf", format="pdf", bbox_inches="tight")
plot.show()

# *****************************************************************************
# plot.subplot(211)
# plot.specgram(SL1_enhanced_speech,Fs=samplingFrequency)
# plot.title(" (c) Enhanced Speech TCN1-FT")
# plot.xlabel('Time')
# plot.ylabel('Frequency')
# plot.show()

# plot.subplot(211)
# plot.specgram(SL3_enhanced_speech,Fs=samplingFrequency)
# plot.title("(d) Enhanced Speech MS-TCN3")
# plot.xlabel('Time')
# plot.ylabel('Frequency')
# plot.show()
# *********************************************************************************


# plot.subplot(211)
# plot.specgram(CH_SL1_enhanced_speech,Fs=samplingFrequency)
# plot.title("(e) Enhanced Speech CWM-TCN1")
# plot.xlabel('Time')
# plot.ylabel('Frequency')
# plot.show()

# plot.subplot(211)
# plot.specgram(CH_SL2_enhanced_speech,Fs=samplingFrequency)
# plot.title("(g) Enhanced Speech CWM-TCN2")
# plot.xlabel('Time')
# plot.ylabel('Frequency')
# plot.show()

plot.subplot(211)
plot.specgram(CH_SL3_enhanced_speech,Fs=samplingFrequency)
plot.title("(C) Enhanced Speech CWM-TCN3")
plot.xlim(0,5)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.savefig("Enhanced_Speech_CWM-TCN3.pdf", format="pdf", bbox_inches="tight")
plot.show()


# /***********************************************
# plot.subplot(211)
# plot.specgram(TCN1_FT_enhanced_speech,Fs=samplingFrequency)
# plot.title("(h) Enhanced Speech TCN1-FT")
# plot.xlabel('Time')
# plot.ylabel('Frequency')
# plot.show()


# plot.subplot(211)
# plot.specgram(TCN2_FT_enhanced_speech,Fs=samplingFrequency)
# plot.title("(e) Enhanced Speech TCN2-FT")
# plot.xlabel('Time')
# plot.ylabel('Frequency')
# plot.show()

 

