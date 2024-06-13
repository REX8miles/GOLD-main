#!/bin/sh
python train_dance_nearest.py --config configs/office-train-config_PDA.yaml --source ./txt/source_amazon_pada.txt --target ./txt/target_caltech_pada.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_PDA.yaml --source ./txt/source_webcam_pada.txt --target ./txt/target_caltech_pada.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_PDA.yaml --source ./txt/source_dslr_pada.txt --target ./txt/target_caltech_pada.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_PDA.yaml --source ./txt/source_dslr_pada.txt --target ./txt/target_amazon_pada.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_PDA.yaml --source ./txt/source_webcam_pada.txt --target ./txt/target_amazon_pada.txt --exp_name "ODA_OOD_LOGIT_END"