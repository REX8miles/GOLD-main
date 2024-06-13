#!/bin/sh
#python train_dance_maths.py --config configs/visda-train-config_ODA.yaml --source ./txt/source_visda_obda.txt --target ./txt/target_visda_obda.txt --exp_name "Maths"
#python train_dance_maths.py --config configs/visda-train-config_UDA.yaml --source ./txt/source_visda_univ.txt --target ./txt/target_visda_univ.txt --exp_name "Maths"
python train_dance_maths.py --config configs/visda-train-config_PDA.yaml --source ./txt/source_visda_pada.txt --target ./txt/target_visda_pada.txt --exp_name "Maths"
python train_dance_maths.py --config configs/visda-train-config_CDA.yaml --source ./txt/source_visda_cls.txt --target ./txt/target_visda_cls.txt --exp_name "Maths"
