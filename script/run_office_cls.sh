#!/bin/sh
python train_dance_nearest.py --config configs/office-train-config_CDA.yaml --source_path ./txt/source_amazon_cls.txt --target_path ./txt/target_dslr_cls.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_CDA.yaml --source_path ./txt/source_amazon_cls.txt --target_path ./txt/target_webcam_cls.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_CDA.yaml --source_path ./txt/source_dslr_cls.txt --target_path ./txt/target_webcam_cls.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_CDA.yaml --source_path ./txt/source_dslr_cls.txt --target_path ./txt/target_amazon_cls.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_CDA.yaml --source_path ./txt/source_webcam_cls.txt --target_path ./txt/target_amazon_cls.txt --exp_name "ODA_OOD_LOGIT_END"
python train_dance_nearest.py --config configs/office-train-config_CDA.yaml --source_path ./txt/source_webcam_cls.txt --target_path ./txt/target_dslr_cls.txt --exp_name "ODA_OOD_LOGIT_END"