#!/usr/bin/env bash
echo 'target_noise test dataset BUS' 
sh gen_scp.sh /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/BUS_Vol3/BUStt_0dB/BUStt_0dB_tt/spk1 /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/BUS_Vol3/BUStt_0dB/BUStt_0dB_tt/target_clean.scp
# sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/BUS_Vol3/BUStt_-3dB/BUStt_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/BUS_Vol3/BUStt_-3dB/BUStt_-3dB_tt/target_noise.scp
echo 'target_noise test dataset PED'
sh gen_scp.sh /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/PED_Vol3/PEDtt_0dB/PEDtt_0dB_tt/spk1 /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/PED_Vol3/PEDtt_0dB/PEDtt_0dB_tt/target_clean.scp
# sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/PED_Vol3/PEDtt_-3dB/PEDtt_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/PED_Vol3/PEDtt_-3dB/PEDtt_-3dB_tt/target_noise.scp
# echo 'target_noise test dataset CAF'
# sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/CAF_Vol3/CAFtt_-3dB/CAFtt_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/CAF_Vol3/CAFtt_-3dB/CAFtt_-3dB_tt/target_clean.scp
# # sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/CAF_Vol3/CAFtt_-3dB/CAFtt_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/CAF_Vol3/CAFtt_-3dB/CAFtt_-3dB_tt/target_noise.scp
echo 'target_noise test dataset STR'
sh gen_scp.sh /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/STR_Vol3/STRtt_0dB/STRtt_0dB_tt/spk1 /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/STR_Vol3/STRtt_0dB/STRtt_0dB_tt/target_clean.scp
# sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/STR_Vol3/STRtt_-3dB/STRtt_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/STR_Vol3/STRtt_-3dB/STRtt_-3dB_tt/target_noise.scp
# # echo 'target_noise test dataset QUT'
# # sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUT_Vol3/QUTttnew_-3dB/QUTttnew_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUT_Vol3/QUTttnew_-3dB/QUTttnew_-3dB_tt/target_clean.scp
# # sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUT_Vol3/QUTttnew_-3dB/QUTttnew_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUT_Vol3/QUTttnew_-3dB/QUTttnew_-3dB_tt/target_noise.scp
echo 'target_noise test dataset QUTSTREET'
sh gen_scp.sh /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTSTREET_Vol3/QUTSTREETttnew_0dB/QUTSTREETttnew_0dB_tt/spk1 /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTSTREET_Vol3/QUTSTREETttnew_0dB/QUTSTREETttnew_0dB_tt/target_clean.scp
# sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUTSTREET_Vol3/QUTSTREETttnew_-3dB/QUTSTREETttnew_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUTSTREET_Vol3/QUTSTREETttnew_-3dB/QUTSTREETttnew_-3dB_tt/target_noise.scp
echo 'target_noise test dataset QUTREVERB'
sh gen_scp.sh /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTREVERB_Vol3/QUTREVERBttnew_0dB/QUTREVERBttnew_0dB_tt/spk1 /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTREVERB_Vol3/QUTREVERBttnew_0dB/QUTREVERBttnew_0dB_tt/target_clean.scp
# sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUTREVERB_Vol3/QUTREVERBttnew_-3dB/QUTREVERBttnew_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUTREVERB_Vol3/QUTREVERBttnew_-3dB/QUTREVERBttnew_-3dB_tt/target_noise.scp
echo 'target_noise test dataset QUTHOME'
sh gen_scp.sh /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTHOMEttnew_0dB/QUTHOMEttnew_0dB_tt/spk1 /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTHOMEttnew_0dB/QUTHOMEttnew_0dB_tt/target_clean.scp
# sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUTHOME_Vol3/QUTHOMEttnew_-3dB/QUTHOMEttnew_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUTHOME_Vol3/QUTHOMEttnew_-3dB/QUTHOMEttnew_-3dB_tt/target_noise.scp
echo 'target_noise test dataset QUTCAR'
sh gen_scp.sh /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTCARttnew_0dB/QUTCARttnew_0dB_tt/spk1 /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTCARttnew_0dB/QUTCARttnew_0dB_tt/target_clean.scp
# sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUTCAR_Vol3/QUTCARttnew_-3dB/QUTCARttnew_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUTCAR_Vol3/QUTCARttnew_-3dB/QUTCARttnew_-3dB_tt/target_noise.scp
echo 'target_noise test dataset QUTCAFE'
sh gen_scp.sh  /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTCAFE_Vol3/QUTCAFEttnew_0dB/QUTCAFEttnew_0dB_tt/spk1 /media/speech70809/Toshibha-3.0TB/MS_SL2_TCN_FUSION/QUTCAFE_Vol3/QUTCAFEttnew_0dB/QUTCAFEttnew_0dB_tt/target_clean.scp
# sh gen_scp.sh /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUTCAFE_Vol3/QUTCAFEttnew_-3dB/QUTCAFEttnew_-3dB_tt/spk1 /media/speech70809/TOSHIBA_3TB1/Channewlwise_R1_SL1_Vol3_saved_models/QUTCAFE_Vol3/QUTCAFEttnew_-3dB/QUTCAFEttnew_-3dB_tt/target_noise.scp
# echo 'denoising testing dataset finished'


