#!/bin/sh
python train_dance_nearest.py --config configs/office-train-config_OPDA.yaml --source ./txt/source_amazon_opda.txt --target ./txt/target_dslr_opda.txt --exp_name "H_score_Experiment"
python train_dance_nearest.py --config configs/office-train-config_OPDA.yaml --source ./txt/source_amazon_opda.txt --target ./txt/target_webcam_opda.txt --exp_name "H_score_Experiment"
python train_dance_nearest.py --config configs/office-train-config_OPDA.yaml --source ./txt/source_dslr_opda.txt --target ./txt/target_webcam_opda.txt --exp_name "H_score_Experiment"
python train_dance_nearest.py --config configs/office-train-config_OPDA.yaml --source ./txt/source_dslr_opda.txt --target ./txt/target_amazon_opda.txt --exp_name "H_score_Experiment"
python train_dance_nearest.py --config configs/office-train-config_OPDA.yaml --source ./txt/source_webcam_opda.txt --target ./txt/target_amazon_opda.txt --exp_name "H_score_Experiment"
python train_dance_nearest.py --config configs/office-train-config_OPDA.yaml --source ./txt/source_webcam_opda.txt --target ./txt/target_dslr_opda.txt --exp_name "H_score_Experiment"