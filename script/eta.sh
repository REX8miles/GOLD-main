#python train_dance_nearest.py --config configs/o_eta0.005.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_webcam_obda.txt --exp_name "eta_experiment"
#python train_dance_nearest.py --config configs/o_eta1.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_webcam_obda.txt --exp_name "eta_experiment"
#python train_dance_nearest.py --config configs/oh_eta0.005.yaml --source ./txt/source_Art_opda.txt --target ./txt/target_Clipart_opda.txt --exp_name "eta_experiment"
python train_dance_nearest.py --config configs/oh_eta1.yaml --source ./txt/source_Art_opda.txt --target ./txt/target_Clipart_opda.txt --exp_name "eta_experiment"
