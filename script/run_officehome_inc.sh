#!/bin/sh
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Real_obda.txt --target_path ./txt/target_Clipart_cls.txt --target_l ./txt/target_Clipart_labeled.txt --exp_name "K/N experiment"
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Real_obda.txt --target_path ./txt/target_Art_cls.txt --target_l ./txt/target_Art_labeled.txt --exp_name "K/N experiment"
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Real_obda.txt --target_path ./txt/target_Product_cls.txt --target_l ./txt/target_Product_labeled.txt --exp_name "K/N experiment"
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Product_obda.txt --target_path ./txt/target_Real_cls.txt --target_l ./txt/target_Real_labeled.txt --exp_name "K/N experiment"
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Product_obda.txt --target_path ./txt/target_Art_cls.txt --target_l ./txt/target_Art_labeled.txt --exp_name "K/N experiment"
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Product_obda.txt --target_path ./txt/target_Clipart_cls.txt --target_l ./txt/target_Clipart_labeled.txt --exp_name "K/N experiment"
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Art_obda.txt --target_path ./txt/target_Clipart_cls.txt --target_l ./txt/target_Clipart_labeled.txt --exp_name "K/N experiment"
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Art_obda.txt --target_path ./txt/target_Product_cls.txt --target_l ./txt/target_Product_labeled.txt --exp_name "K/N experiment"
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Art_obda.txt --target_path ./txt/target_Real_cls.txt --target_l ./txt/target_Real_labeled.txt --exp_name "K/N experiment"
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Clipart_obda.txt --target_path ./txt/target_Real_cls.txt --target_l ./txt/target_Real_labeled.txt --exp_name "K/N experiment"
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Clipart_obda.txt --target_path ./txt/target_Art_cls.txt --target_l ./txt/target_Art_labeled.txt --exp_name "K/N experiment"
python train_dance_novel.py --config configs/officehome-train-config_CLDA.yaml --source ./txt/source_Clipart_obda.txt --target_path ./txt/target_Product_cls.txt --target_l ./txt/target_Product_labeled.txt --exp_name "K/N experiment"

