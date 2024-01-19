#!/usr/bin/env bash
# echo 'denoise test dataset BUS'     
# python nnet/separate_new.py //media/speech70809/Data02/MS_SL2_split_256channelWise_w4_16batch_models --input  /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_-3dB/mix.scp --gpu 0 --fs 16000 --dump-dir /media/speech70809/Data02/MS_SL2_split_256channelWise/BUS_Vol3/BUStt_-3dB/BUStt_-3dB_tt
# echo 'denoise test dataset PED'
# python nnet/separate_new.py /media/speech70809/Toshibha-3.0TB/MS_SL2_splite_fusion_expo_group --input /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/PED_Vol3/PEDtt_-3dB/mix.scp --gpu 0 --fs 16000 --dump-dir  /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/PED_Vol3/PEDtt_-3dB/PEDtt_-3dB_tt
# echo 'denoise test dataset STR'
# python nnet/separate_new.py /media/speech70809/Toshibha-3.0TB/MS_SL2_splite_fusion_expo_group --input /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/STR_Vol3/STRtt_-3dB/mix.scp --gpu 0 --fs 16000 --dump-dir  /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/STR_Vol3/STRtt_-3dB/STRtt_-3dB_tt

# echo 'denoise test dataset QUTSTREET'
# python nnet/separate_new.py /media/speech70809/Toshibha-3.0TB/MS_SL2_splite_fusion_expo_group --input /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTSTREET_Vol3/QUTSTREETttnew_-3dB/mix.scp --gpu 0 --fs 16000 --dump-dir  /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTSTREET_Vol3/QUTSTREETttnew_-3dB/QUTSTREETttnew_-3dB_tt
# echo 'denoise test dataset QUTREVERB'
# python nnet/separate_new.py /media/speech70809/Toshibha-3.0TB/MS_SL2_splite_fusion_expo_group --input /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTREVERB_Vol3/QUTREVERBttnew_-3dB/mix.scp --gpu 0 --fs 16000 --dump-dir  /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTREVERB_Vol3/QUTREVERBttnew_-3dB/QUTREVERBttnew_-3dB_tt
# echo 'denoise test dataset QUTHOME'
# python nnet/separate_new.py /media/speech70809/Toshibha-3.0TB/MS_SL2_splite_fusion_expo_group --input /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTHOME_Vol3/QUTHOMEttnew_-3dB/mix.scp --gpu 0 --fs 16000 --dump-dir  /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTHOME_Vol3/QUTHOMEttnew_-3dB/QUTHOMEttnew_-3dB_tt
# echo 'denoise test dataset QUTCAR'
# python nnet/separate_new.py /media/speech70809/Toshibha-3.0TB/MS_SL2_splite_fusion_expo_group --input /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAR_Vol3/QUTCARttnew_-3dB/mix.scp --gpu 0 --fs 16000 --dump-dir  /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTCAR_Vol3/QUTCARttnew_-3dB/QUTCARttnew_-3dB_tt
# echo 'denoise test dataset QUTCAFE'
# python nnet/separate_new.py /media/speech70809/Toshibha-3.0TB/MS_SL2_splite_fusion_expo_group --input /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAFE_Vol3/QUTCAFEttnew_-3dB/mix.scp --gpu 0 --fs 16000 --dump-dir  /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTCAFE_Vol3/QUTCAFEttnew_-3dB/QUTCAFEttnew_-3dB_tt
# echo 'denoising testing dataset finished'


echo 'Speech Denoiseing of Far Eastern Hospital'
python nnet/separate_new.py /media/speech70809/Data022/MS_SL3_split_256channelWise_w4_16batch_models --input /media/speech70809/Data011/speech_donoiser_new/datasets/Far_Eastern_Hosp/Dataset_16k.scp --gpu 0 --fs 16000 --dump-dir /media/speech70809/Data022/Far_Eastern_Hosp

