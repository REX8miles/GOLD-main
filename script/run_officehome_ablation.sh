python train_dance_Novat.py \
      --config configs/officehome-train-config_ODA.yaml \
      --source ./txt/source_Art_obda.txt \
      --target ./txt/target_Product_obda.txt \
      --exp_name "Ablation_Experimen/Novat"

python train_dance_NoOOD.py \
      --config configs/officehome-train-config_ODA.yaml \
      --source ./txt/source_Art_obda.txt \
      --target ./txt/target_Product_obda.txt \
      --exp_name "Ablation_Experimen/NoOOD"

python train_dance_NoSelfAttention.py \
      --config configs/officehome-train-config_ODA.yaml \
      --source ./txt/source_Art_obda.txt \
      --target ./txt/target_Product_obda.txt \
      --exp_name "Ablation_Experimen/NoSelfAttention"

python train_dance_NoCon.py \
      --config configs/officehome-train-config_ODA.yaml \
      --source ./txt/source_Art_obda.txt \
      --target ./txt/target_Product_obda.txt \
      --exp_name "Ablation_Experimen/NoCon"