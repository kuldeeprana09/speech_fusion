#!/usr/bin/env bash

noise_file=$1
echo $noise_file
if [ -f $noise_file ]; then
    echo 'rm existed '$noise_file
    rm  $noise_file
fi
echo 'generate musan list'
sh gen_noisefiles.sh $noise_file music /media/lab70809/Data01/speech_donoiser_new/datasets/musan/music
sh gen_noisefiles.sh $noise_file noise /media/lab70809/Data01/speech_donoiser_new/datasets/musan/noise
echo 'generate chime4 list'
sh gen_noisefiles.sh $noise_file BUS /media/lab70809/Data01/speech_donoiser_new/datasets/chime4_noise_classified/BUS
sh gen_noisefiles.sh $noise_file CAF /media/lab70809/Data01/speech_donoiser_new/datasets/chime4_noise_classified/CAF
sh gen_noisefiles.sh $noise_file PED /media/lab70809/Data01/speech_donoiser_new/datasets/chime4_noise_classified/PED
sh gen_noisefiles.sh $noise_file STR /media/lab70809/Data01/speech_donoiser_new/datasets/chime4_noise_classified/STR
echo 'generate wham list'
sh gen_noisefiles.sh $noise_file wham /media/lab70809/Data01/speech_donoiser_new/datasets/wham_noise
echo 'concat DNS list'
cat noise_types_16k_Freesound.csv >> $noise_file
cat noise_types_16k_audioset.csv >> $noise_file