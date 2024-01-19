#!/usr/bin/env bash

python getDNSaudiosetCsv.py --noisedir /media/lab70809/Data01/speech_donoiser_new/datasets/noise/
python getDNSfreesoundCsv.py --noisedir /media/lab70809/Data01/speech_donoiser_new/datasets/noise --csvdir /media/lab70809/Data01/speech_donoiser_new/datasets
sh gen_noise_csv.sh /media/lab70809/Data01/speech_donoiser_new/datasets/noise.csv
python train_test_split.py --noise /media/lab70809/Data01/speech_donoiser_new/datasets/noise.csv