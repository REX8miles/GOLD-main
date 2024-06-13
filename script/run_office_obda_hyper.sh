#!/bin/sh
python train_dance_nearest.py --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_dslr_obda.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_webcam_obda.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_ODA.yaml --source ./txt/source_dslr_obda.txt --target ./txt/target_webcam_obda.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_ODA.yaml --source ./txt/source_dslr_obda.txt --target ./txt/target_amazon_obda.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_ODA.yaml --source ./txt/source_webcam_obda.txt --target ./txt/target_amazon_obda.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_ODA.yaml --source ./txt/source_webcam_obda.txt --target ./txt/target_dslr_obda.txt --exp_name "ODA_OOD_LOGIT_END"