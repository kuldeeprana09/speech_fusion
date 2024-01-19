#!/usr/bin/env bash

#sh main/tools/gen_scp_all.sh /media/lab70809/Data02/speech_donoiser_new/datasets/ner-300hr /media/lab70809/Data02/speech_donoiser_new/main
python main/nnet/trainnew.py --gpus 0 --checkpoint /media/lab70809/Data02/speech_donoiser_new/saved_models --batch-size 2
